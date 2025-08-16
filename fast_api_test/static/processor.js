class HitDetectorProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0][0]; // first channel
    if (!input) return true;

    const max = Math.max(...input.map(Math.abs));
    if (max > 0.5) {
      console.log("Hit detected!", max);
    }

    return true; // keep alive
  }
}

registerProcessor('hit-detector', HitDetectorProcessor);
