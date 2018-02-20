require_relative 'neuron'

class Layer
  attr_reader :neurons

  def initialize(neurons_count, inputs_count)
    @neurons = Array.new(neurons_count)
    @neurons.map! { Neuron.new(inputs_count) }
  end

  def process(inputs)
    @neurons.map do |neuron|
      neuron.activate(inputs)
    end
  end
end
