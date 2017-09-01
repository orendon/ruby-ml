require_relative 'math_utils'

class Neuron
  attr_accessor :weights,
                :bias,
                :error,       # error gap during back prop
                :last_output, # last neuron ouput
                :last_inputs, # last neuron inputs
                :delta,       # delta step, derivative for gradient descend
                :errors       # errors per training sample

  def initialize(inputs_count)
    @weights = Array.new(inputs_count)
    @weights.map! { MathUtils.random }

    @bias = MathUtils.random
  end

  def activate(inputs)
    @last_inputs = inputs
    sum = 0

    # perceptron (dot product)
    inputs.each_with_index do |input, idx|
      sum += input * @weights[idx]
    end
    sum += @bias

    # activation function
    @last_output = MathUtils.sigmoid(sum)
  end
end
