"""
    Drift

Concept drift detection algorithms.

# Exports
- `ADWIN` - Adaptive Windowing for drift detection
- `DDM` - Drift Detection Method
- `EDDM` - Early Drift Detection Method
- `PageHinkley` - Page-Hinkley Test
- `KSWIN` - Kolmogorov-Smirnov Windowed drift detector
"""
module Drift

using OnlineStatsBase
using ..OnlineML: Detector, DriftStatus, NoDrift, Warning, DriftDetected
import ..OnlineML: status, reset!, update!, detected_drift, detected_warning

include("adwin.jl")
include("ddm.jl")
include("page_hinkley.jl")
include("kswin.jl")
include("wrapper.jl")

export ADWIN, DDM, EDDM, PageHinkley, KSWIN
export DriftRetrainingWrapper, drift_count, warning_count, detector_status

end # module Drift
