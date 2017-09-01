require_relative 'layer'

class MLP
  attr_reader :layers

  def initialize
    @layers = Array.new
  end

  def add_layer(neurons_count, inputs_count)
    @layers << Layer.new(neurons_count, inputs_count)
  end

  def predict(inputs)
    feed_forward(inputs)
  end

  def feed_forward(inputs)
    output = 0
    @layers.each do |layer|
      output = layer.process(inputs)
      inputs = output
    end

    output
  end
end
