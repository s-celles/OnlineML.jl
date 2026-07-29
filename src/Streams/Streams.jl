"""
    Streams

Data stream utilities and synthetic generators.

# Exports
- `DataStream` - Iterator wrapper for data streams
- `SineGenerator` - Synthetic sine wave stream
- `RandomRBFGenerator` - Random RBF-based stream
- `ConceptDriftStream` - Stream with concept drift
- `LEDGenerator` - LED digit display stream
- `AgrawalGenerator` - Agrawal demographic stream
- `SEAGenerator` - SEA concepts stream
- `HyperplaneGenerator` - Rotating hyperplane stream
"""
module Streams

using Random

include("generators.jl")
include("iterators.jl")

export DataStream, SineGenerator, RandomRBFGenerator, ConceptDriftStream
export LEDGenerator, AgrawalGenerator, SEAGenerator, HyperplaneGenerator
export iterate_stream, generate
export batch_stream, take_stream, skip_stream
export BatchedStream, TakeStream, SkipStream

end # module Streams
