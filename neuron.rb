require_relative 'math_utils'

class Neuron
  attr_reader :weights, :bias

  def initialize(inputs_count)
    @weights = Array.new(inputs_count)
    @weights.map! { MathUtils.random }

    @bias = MathUtils.random
  end

  def activate(inputs)
    sum = 0

    # perceptron (dot product)
    inputs.each_with_index do |input, idx|
      sum += input * @weights[idx]
    end
    sum += bias

    # activation function
    MathUtils.sigmoid(sum)
  end
end
