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
	testHostGPUV2AnomalyMetric   = "hami_host_gpu_memory_v2_anomaly"
	testHostGPUUtilizationMetric = "hami_host_gpu_utilization_ratio"
)

// Byte values modelled on the GPU5 TITAN RTX cards, where the NVML v2 identity
// Total = Free + Used + Reserved was confirmed against nvidia-smi.
const (
	testGPUTotalBytes        = 24576 * 1024 * 1024 // 24 GiB
	testDriverReservedBytes  = 554 * 1024 * 1024   // driver reservation, ~0.5413 GiB
	testIdleAllocatedBytes   = 9 * 1024 * 1024     // idle card process allocation
	testLoadedAllocatedBytes = 8 * 1024 * 1024 * 1024
	testIdleCombinedBytes    = testIdleAllocatedBytes + testDriverReservedBytes   // 563 MiB
	testLoadedCombinedBytes  = testLoadedAllocatedBytes + testDriverReservedBytes // 8.54 GiB
)

type capturedMemoryMetric struct {
	name   string
	value  float64
	labels map[string]string
}

// TestCollectGPUMemoryMetricsUsesOneV2SnapshotForCombinedAndSplit pins the NVML v2
// semantics Total = Free + Used + Reserved. v2 Used is the process allocation and
// EXCLUDES the driver reservation, so allocated is v2 Used verbatim and the combined
// gauge adds the reservation back to keep one definition (the v1 one) across modes.
func TestCollectGPUMemoryMetricsUsesOneV2SnapshotForCombinedAndSplit(t *testing.T) {
	for _, tc := range []struct {
		name         string
		used         uint64
		wantCombined float64
	}{
		{
			// The idle state that used to trip the old used < reserved guard on
			// every scrape: 9 MiB allocated, 554 MiB reserved, 563 MiB combined.
			name:         "idle card allocates less than the driver reservation",
			used:         testIdleAllocatedBytes,
			wantCombined: testIdleCombinedBytes,
		},
		{
			name:         "loaded card allocates more than the driver reservation",
			used:         testLoadedAllocatedBytes,
			wantCombined: testLoadedCombinedBytes,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// The v1 value deliberately differs from v2. A v2 split must keep the
			// compatible combined value on the same v2 snapshot as reserved/allocated.
			metrics, err := collectGPUMemoryMetricsForTest(t, 128, nvml.SUCCESS, nvml.Memory_v2{
				Total:    testGPUTotalBytes,
				Used:     tc.used,
				Reserved: testDriverReservedBytes,
			})
			if err != nil {
				t.Fatalf("collectGPUMemoryMetrics() error = %v", err)
			}

			assertMetricValue(t, metrics, testHostGPUUsedMetric, tc.wantCombined)
			assertMetricValue(t, metrics, testHostGPUReservedMetric, testDriverReservedBytes)
			assertMetricValue(t, metrics, testHostGPUAllocatedMetric, float64(tc.used))
			assertMetricMode(t, metrics, gpuMemoryAccountingModeV2Split)
			// A genuine v2 split is not an anomaly.
			assertMetricValue(t, metrics, testHostGPUV2AnomalyMetric, 0)
			assertSharedDeviceLabels(t, metrics)
		})
	}
}

// TestClassifyGPUMemoryAccountingAcceptsUsedBelowReserved is the direct regression test
// for the accounting-mode oscillation: an idle card reports v2 Used below Reserved on
// every scrape, and that must classify as a stable v2_split rather than an anomaly.
func TestClassifyGPUMemoryAccountingAcceptsUsedBelowReserved(t *testing.T) {
	accounting, err := classifyGPUMemoryAccounting(nvml.Memory_v2{
		Total:    testGPUTotalBytes,
		Used:     testIdleAllocatedBytes,
		Reserved: testDriverReservedBytes,
	}, nvml.SUCCESS, true)
	if err != nil {
		t.Fatalf("classifyGPUMemoryAccounting() error = %v", err)
	}
	if accounting.mode != gpuMemoryAccountingModeV2Split {
		t.Errorf("mode = %q, want %q", accounting.mode, gpuMemoryAccountingModeV2Split)
	}
	if !accounting.hasSplit {
		t.Error("hasSplit = false, want true")
	}
	if accounting.v2Anomaly {
		t.Error("v2Anomaly = true, want false")
	}
	if accounting.allocated != testIdleAllocatedBytes {
		t.Errorf("allocated = %d, want %d", accounting.allocated, uint64(testIdleAllocatedBytes))
	}
	if accounting.reserved != testDriverReservedBytes {
		t.Errorf("reserved = %d, want %d", accounting.reserved, uint64(testDriverReservedBytes))
	}
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
			// v2 simply does not exist on such a host; that is not an anomaly.
			assertMetricValue(t, metrics, testHostGPUV2AnomalyMetric, 0)
			assertMetricAbsent(t, metrics, testHostGPUReservedMetric)
			assertMetricAbsent(t, metrics, testHostGPUAllocatedMetric)
		})
	}
}

// TestCollectGPUMemoryMetricsLabelsUnusableV2AsV1Combined covers the cases where the v2
// query cannot produce a trustworthy split. The value served is the v1 combined reading,
// so the mode label must say v1_combined rather than unknown; the anomaly gauge, not the
// mode label, carries the "this driver's v2 path is misbehaving" signal.
func TestCollectGPUMemoryMetricsLabelsUnusableV2AsV1Combined(t *testing.T) {
	for _, tc := range []struct {
		name        string
		ret         nvml.Return
		memory      nvml.Memory_v2
		wantAnomaly float64
	}{
		{
			name:        "v2 error",
			ret:         nvml.ERROR_UNKNOWN,
			wantAnomaly: 0,
		},
		{
			// Breaks Total = Free + Used + Reserved, so no split can be derived.
			name: "used plus reserved exceeds total",
			ret:  nvml.SUCCESS,
			memory: nvml.Memory_v2{
				Total:    testGPUTotalBytes,
				Used:     testGPUTotalBytes,
				Reserved: testDriverReservedBytes,
			},
			wantAnomaly: 1,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			metrics, err := collectGPUMemoryMetricsForTest(t, 128, tc.ret, tc.memory)
			if err != nil {
				t.Fatalf("collectGPUMemoryMetrics() error = %v", err)
			}

			assertMetricValue(t, metrics, testHostGPUUsedMetric, 128)
			assertMetricMode(t, metrics, gpuMemoryAccountingModeV1Combined)
			assertMetricValue(t, metrics, testHostGPUV2AnomalyMetric, tc.wantAnomaly)
			// No split is fabricated from an untrustworthy v2 snapshot.
			assertMetricAbsent(t, metrics, testHostGPUReservedMetric)
			assertMetricAbsent(t, metrics, testHostGPUAllocatedMetric)
		})
	}
}

// TestClassifyGPUMemoryAccountingReportsUnknownWithoutAnySource pins the one remaining
// use of the unknown mode: neither v1 nor v2 produced a value. collectGPUMemoryMetrics
// returns early when the v1 call fails, so this state is only reachable at the
// classifier boundary.
func TestClassifyGPUMemoryAccountingReportsUnknownWithoutAnySource(t *testing.T) {
	for _, tc := range []struct {
		name        string
		ret         nvml.Return
		memory      nvml.Memory_v2
		wantAnomaly bool
	}{
		{
			name: "v2 not supported",
			ret:  nvml.ERROR_NOT_SUPPORTED,
		},
		{
			name: "v2 error",
			ret:  nvml.ERROR_UNKNOWN,
		},
		{
			name:        "used plus reserved exceeds total",
			ret:         nvml.SUCCESS,
			memory:      nvml.Memory_v2{Total: testGPUTotalBytes, Used: testGPUTotalBytes, Reserved: testDriverReservedBytes},
			wantAnomaly: true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			accounting, _ := classifyGPUMemoryAccounting(tc.memory, tc.ret, false)
			if accounting.mode != gpuMemoryAccountingModeUnknown {
				t.Errorf("mode = %q, want %q", accounting.mode, gpuMemoryAccountingModeUnknown)
			}
			if accounting.hasSplit {
				t.Error("hasSplit = true, want false")
			}
			if accounting.v2Anomaly != tc.wantAnomaly {
				t.Errorf("v2Anomaly = %v, want %v", accounting.v2Anomaly, tc.wantAnomaly)
			}
		})
	}
}

func TestCollectGPUDeviceMetricsContinuesUtilizationWhenV2AccountingIsUnusable(t *testing.T) {
	device := testNVMLDevice(128, nvml.ERROR_UNKNOWN, nvml.Memory_v2{})
	originalDeviceGetHandleByIndex := deviceGetHandleByIndex
	deviceGetHandleByIndex = func(int) (nvml.Device, nvml.Return) {
		return device, nvml.SUCCESS
	}
	t.Cleanup(func() {
		deviceGetHandleByIndex = originalDeviceGetHandleByIndex
	})

	ch := make(chan prometheus.Metric, 8)
	err := (ClusterManagerCollector{}).collectGPUDeviceMetrics(ch, 0)
	close(ch)
	if err != nil {
		t.Fatalf("collectGPUDeviceMetrics() error = %v", err)
	}

	metrics := captureMetricsForTest(t, ch)
	assertMetricValue(t, metrics, testHostGPUUsedMetric, 128)
	assertMetricMode(t, metrics, gpuMemoryAccountingModeV1Combined)
	assertMetricValue(t, metrics, testHostGPUV2AnomalyMetric, 0)
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
	ch := make(chan prometheus.Metric, 8)
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
		testHostGPUV2AnomalyMetric,
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
	// The anomaly gauge is labelled exactly like the mode-info metric, so both must
	// agree on the mode.
	for _, name := range []string{testHostGPUModeMetric, testHostGPUV2AnomalyMetric} {
		if got := findMetric(t, metrics, name).labels["memory_accounting_mode"]; got != string(want) {
			t.Errorf("metric %q memory accounting mode = %q, want %q", name, got, want)
		}
	}
}

func assertSharedDeviceLabels(t *testing.T, metrics map[string]capturedMemoryMetric) {
	t.Helper()
	used := findMetric(t, metrics, testHostGPUUsedMetric).labels
	for _, name := range []string{testHostGPUReservedMetric, testHostGPUAllocatedMetric, testHostGPUModeMetric, testHostGPUV2AnomalyMetric} {
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
