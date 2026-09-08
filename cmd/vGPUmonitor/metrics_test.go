/*
Copyright 2026 The HAMi Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"sort"
	"strings"
	"testing"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	nvmlmock "github.com/NVIDIA/go-nvml/pkg/nvml/mock"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

const (
	testHostGPUUsedMetric        = "hami_host_gpu_memory_used_bytes"
	testHostGPUReservedMetric    = "hami_host_gpu_memory_reserved_bytes"
	testHostGPUAllocatedMetric   = "hami_host_gpu_memory_allocated_bytes"
	testHostGPUModeMetric        = "hami_host_gpu_memory_accounting_mode_info"
	testHostGPUUtilizationMetric = "hami_host_gpu_utilization_ratio"
)

type capturedMemoryMetric struct {
	name   string
	value  float64
	labels map[string]string
}

func TestCollectGPUMemoryMetricsUsesOneV2SnapshotForCombinedAndSplit(t *testing.T) {
	metrics, err := collectGPUMemoryMetricsForTest(t, 128, nvml.SUCCESS, nvml.Memory_v2{
		Used:     160,
		Reserved: 32,
	})
	if err != nil {
		t.Fatalf("collectGPUMemoryMetrics() error = %v", err)
	}

	// The v1 value deliberately differs from v2. A v2 split must keep the
	// compatible combined value on the same v2 snapshot as reserved/allocated.
	assertMetricValue(t, metrics, testHostGPUUsedMetric, 160)
	assertMetricValue(t, metrics, testHostGPUReservedMetric, 32)
	assertMetricValue(t, metrics, testHostGPUAllocatedMetric, 128)
	assertMetricMode(t, metrics, gpuMemoryAccountingModeV2Split)
	assertSharedDeviceLabels(t, metrics)
}

func TestCollectGPUMemoryMetricsUsesV1CombinedFallbackWithoutSplit(t *testing.T) {
	for _, ret := range []nvml.Return{nvml.ERROR_NOT_SUPPORTED, nvml.ERROR_FUNCTION_NOT_FOUND} {
		t.Run(nvml.ErrorString(ret), func(t *testing.T) {
			metrics, err := collectGPUMemoryMetricsForTest(t, 128, ret, nvml.Memory_v2{})
			if err != nil {
				t.Fatalf("collectGPUMemoryMetrics() error = %v", err)
			}

			assertMetricValue(t, metrics, testHostGPUUsedMetric, 128)
			assertMetricMode(t, metrics, gpuMemoryAccountingModeV1Combined)
			assertMetricAbsent(t, metrics, testHostGPUReservedMetric)
			assertMetricAbsent(t, metrics, testHostGPUAllocatedMetric)
		})
	}
}

func TestCollectGPUMemoryMetricsReportsUnknownWithoutFabricatedSplit(t *testing.T) {
	for _, tc := range []struct {
		name   string
		ret    nvml.Return
		memory nvml.Memory_v2
	}{
		{
			name: "v2 error",
			ret:  nvml.ERROR_UNKNOWN,
		},
		{
			name: "reserved exceeds used",
			ret:  nvml.SUCCESS,
			memory: nvml.Memory_v2{
				Used:     31,
				Reserved: 32,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			metrics, err := collectGPUMemoryMetricsForTest(t, 128, tc.ret, tc.memory)
			if err != nil {
				t.Fatalf("collectGPUMemoryMetrics() error = %v", err)
			}

			assertMetricValue(t, metrics, testHostGPUUsedMetric, 128)
			assertMetricMode(t, metrics, gpuMemoryAccountingModeUnknown)
			assertMetricAbsent(t, metrics, testHostGPUReservedMetric)
			assertMetricAbsent(t, metrics, testHostGPUAllocatedMetric)
		})
	}
}

func TestCollectGPUDeviceMetricsContinuesUtilizationWhenV2AccountingIsUnknown(t *testing.T) {
	device := testNVMLDevice(128, nvml.ERROR_UNKNOWN, nvml.Memory_v2{})
	originalDeviceGetHandleByIndex := deviceGetHandleByIndex
	deviceGetHandleByIndex = func(int) (nvml.Device, nvml.Return) {
		return device, nvml.SUCCESS
	}
	t.Cleanup(func() {
		deviceGetHandleByIndex = originalDeviceGetHandleByIndex
	})

	ch := make(chan prometheus.Metric, 5)
	err := (ClusterManagerCollector{}).collectGPUDeviceMetrics(ch, 0)
	close(ch)
	if err != nil {
		t.Fatalf("collectGPUDeviceMetrics() error = %v", err)
	}

	metrics := captureMetricsForTest(t, ch)
	assertMetricValue(t, metrics, testHostGPUUsedMetric, 128)
	assertMetricMode(t, metrics, gpuMemoryAccountingModeUnknown)
	assertMetricAbsent(t, metrics, testHostGPUReservedMetric)
	assertMetricAbsent(t, metrics, testHostGPUAllocatedMetric)
	assertMetricValue(t, metrics, testHostGPUUtilizationMetric, 42)
}

func TestCollectGPUDeviceMetricsPreservesV1MemoryErrorPath(t *testing.T) {
	device := &nvmlmock.Device{
		GetMemoryInfoFunc: func() (nvml.Memory, nvml.Return) {
			return nvml.Memory{}, nvml.ERROR_UNKNOWN
		},
	}
	originalDeviceGetHandleByIndex := deviceGetHandleByIndex
	deviceGetHandleByIndex = func(int) (nvml.Device, nvml.Return) {
		return device, nvml.SUCCESS
	}
	t.Cleanup(func() {
		deviceGetHandleByIndex = originalDeviceGetHandleByIndex
	})

	ch := make(chan prometheus.Metric, 1)
	err := (ClusterManagerCollector{}).collectGPUDeviceMetrics(ch, 0)
	close(ch)
	if err == nil {
		t.Fatal("collectGPUDeviceMetrics() error = nil, want v1 memory error")
	}
	if metrics := captureMetricsForTest(t, ch); len(metrics) != 0 {
		t.Fatalf("metric series count = %d, want 0", len(metrics))
	}
}

func collectGPUMemoryMetricsForTest(t *testing.T, v1Used uint64, v2Ret nvml.Return, v2Memory nvml.Memory_v2) (map[string]capturedMemoryMetric, error) {
	t.Helper()
	ch := make(chan prometheus.Metric, 4)
	err := (ClusterManagerCollector{}).collectGPUMemoryMetrics(ch, testNVMLDevice(v1Used, v2Ret, v2Memory), 0)
	close(ch)
	return captureMetricsForTest(t, ch), err
}

func testNVMLDevice(v1Used uint64, v2Ret nvml.Return, v2Memory nvml.Memory_v2) *nvmlmock.Device {
	return &nvmlmock.Device{
		GetMemoryInfoFunc: func() (nvml.Memory, nvml.Return) {
			return nvml.Memory{Used: v1Used}, nvml.SUCCESS
		},
		GetMemoryInfo_v2Func: func() (nvml.Memory_v2, nvml.Return) {
			return v2Memory, v2Ret
		},
		GetUUIDFunc: func() (string, nvml.Return) {
			return "unit-test-device", nvml.SUCCESS
		},
		GetNameFunc: func() (string, nvml.Return) {
			return "unit-test", nvml.SUCCESS
		},
		GetUtilizationRatesFunc: func() (nvml.Utilization, nvml.Return) {
			return nvml.Utilization{Gpu: 42}, nvml.SUCCESS
		},
	}
}

func captureMetricsForTest(t *testing.T, ch <-chan prometheus.Metric) map[string]capturedMemoryMetric {
	t.Helper()
	metrics := make(map[string]capturedMemoryMetric)
	for metric := range ch {
		name := hostGPUMemoryMetricName(t, metric)
		protoMetric := &dto.Metric{}
		if err := metric.Write(protoMetric); err != nil {
			t.Fatalf("metric.Write() error = %v", err)
		}
		labels := make(map[string]string, len(protoMetric.Label))
		for _, label := range protoMetric.Label {
			labels[label.GetName()] = label.GetValue()
		}
		key := metricSeriesKey(name, labels)
		if _, found := metrics[key]; found {
			t.Fatalf("duplicate metric series %q", key)
		}
		metrics[key] = capturedMemoryMetric{
			name:   name,
			value:  protoMetric.GetGauge().GetValue(),
			labels: labels,
		}
	}
	return metrics
}

func hostGPUMemoryMetricName(t *testing.T, metric prometheus.Metric) string {
	t.Helper()
	desc := metric.Desc().String()
	for _, name := range []string{
		testHostGPUUsedMetric,
		testHostGPUReservedMetric,
		testHostGPUAllocatedMetric,
		testHostGPUModeMetric,
		testHostGPUUtilizationMetric,
	} {
		if strings.Contains(desc, `fqName: "`+name+`"`) {
			return name
		}
	}
	t.Fatalf("unexpected metric descriptor: %s", desc)
	return ""
}

func metricSeriesKey(name string, labels map[string]string) string {
	parts := make([]string, 0, len(labels))
	for labelName, labelValue := range labels {
		parts = append(parts, labelName+"="+labelValue)
	}
	sort.Strings(parts)
	return name + "{" + strings.Join(parts, ",") + "}"
}

func assertMetricValue(t *testing.T, metrics map[string]capturedMemoryMetric, name string, want float64) {
	t.Helper()
	metric := findMetric(t, metrics, name)
	if metric.value != want {
		t.Errorf("metric %q value = %v, want %v", name, metric.value, want)
	}
}

func assertMetricAbsent(t *testing.T, metrics map[string]capturedMemoryMetric, name string) {
	t.Helper()
	for _, metric := range metrics {
		if metric.name == name {
			t.Errorf("metric %q was emitted", name)
			return
		}
	}
}

func assertMetricMode(t *testing.T, metrics map[string]capturedMemoryMetric, want gpuMemoryAccountingMode) {
	t.Helper()
	assertMetricValue(t, metrics, testHostGPUModeMetric, 1)
	if got := findMetric(t, metrics, testHostGPUModeMetric).labels["memory_accounting_mode"]; got != string(want) {
		t.Errorf("memory accounting mode = %q, want %q", got, want)
	}
}

func assertSharedDeviceLabels(t *testing.T, metrics map[string]capturedMemoryMetric) {
	t.Helper()
	used := findMetric(t, metrics, testHostGPUUsedMetric).labels
	for _, name := range []string{testHostGPUReservedMetric, testHostGPUAllocatedMetric, testHostGPUModeMetric} {
		for _, labelName := range []string{"device_index", "device_uuid", "device_type"} {
			if got := findMetric(t, metrics, name).labels[labelName]; got != used[labelName] {
				t.Errorf("metric %q label %q = %q, want %q", name, labelName, got, used[labelName])
			}
		}
	}
}

func findMetric(t *testing.T, metrics map[string]capturedMemoryMetric, name string) capturedMemoryMetric {
	t.Helper()
	var matches []capturedMemoryMetric
	for _, metric := range metrics {
		if metric.name == name {
			matches = append(matches, metric)
		}
	}
	if len(matches) != 1 {
		t.Fatalf("metric %q series count = %d, want 1", name, len(matches))
	}
	return matches[0]
}
