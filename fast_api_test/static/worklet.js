class HitDetectorProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = [];
        this.postBuffer = [];
        this.collectingPost = false;
        this.threshold = 0.5;
        this.preTriggerSamples = 4410;
        this.postTriggerSamples = 4410;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0][0]; // mono

        for (let i = 0; i < input.length; i++) {
            const sample = input[i];
            this.buffer.push(sample);

            if (this.collectingPost) {
                this.postBuffer.push(sample);
                if (this.postBuffer.length >= this.postTriggerSamples) {
                    const snippet = this.buffer.slice(-this.preTriggerSamples).concat(this.postBuffer);
                    this.port.postMessage({ type: 'hit', snippet });
                    this.collectingPost = false;
                    this.postBuffer = [];
                }
            } else if (Math.abs(sample) > this.threshold) {
                this.collectingPost = true;
                this.postBuffer = [];
            }

            if (this.buffer.length > this.preTriggerSamples) this.buffer.shift();
        }

        return true; // keep processor alive
    }
}

registerProcessor('hit-detector', HitDetectorProcessor);
