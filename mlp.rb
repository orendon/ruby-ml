require_relative 'layer'
require_relative 'math_utils'

class MLP
  attr_reader :layers

  def initialize
    @layers = Array.new
    @error_threshold = 0.00001
    @learning_rate = 0.3
  end

  def add_layer(neurons_count, inputs_count)
    @layers << Layer.new(neurons_count, inputs_count)
  end

  # feed forward
  def predict(inputs)
    output = 0
    @layers.each do |layer|
      output = layer.process(inputs)
      inputs = output
    end

    output
  end

  def printy
    @layers.each do |l|
      l.neurons.each do |n|
        #printf "li: %s lo: %s \n", n.last_inputs, n.last_output
        printf "--------- \n"
        #printf "ws: %s \n", n.weights
        printf "b: %s err: %s delta: %s \n", n.bias, n.error, n.delta
      end
    end
  end

  def learn(training_data)
    output_layer = @layers.last

    #500_000.times do |i|
    20.times do |i|
      printy

      @iter = i
      back_propagation([training_data.first], output_layer)
    end
  end

  def back_propagation(training_data, output_layer)
    training_data.each_with_index do |sample, sx|
      data, target = sample
      output = predict(data)

      # calculate from output layer
      # since all layers depend on end result
      output_layer.neurons.each_with_index do |neuron, idx|
        neuron.error = target[idx] - output[idx] # gap difference
        neuron.delta = MathUtils.sigmoid_derivative(neuron.last_output) * neuron.error

        # track errors for each training sample
        neuron.errors = neuron.errors || []
        neuron.errors[sx] = neuron.error
      end

      # exclude output layer from propagation
      (@layers.size-2).downto(0).each do |l|
        curr_layer = @layers[l]
        next_layer = @layers[l+1]

        curr_layer.neurons.each_with_index do |neuron, idx|
          neuron.error = next_layer.neurons.map { |neu| neu.weights[idx] * neu.delta }.reduce(:+)
          neuron.delta = MathUtils.sigmoid_derivative(neuron.last_output) * neuron.error

          # update weight and bias
          next_layer.neurons.each do |neu|
            neu.weights.each_with_index do |weight, w|
              weight += @learning_rate * neu.last_inputs[w] * neu.delta
            end
            neu.bias = @learning_rate * neu.delta
          end # update w & b

        end # update error & delta
      end # propagate

    end # training data

    training_errors = output_layer.neurons.reduce([]){ |errors, neu| errors.concat(neu.errors) }
    mse = MathUtils.mse(training_errors)

    puts "iter: #{@iter}, mse: #{mse}" #if @iter % 10_000 == 0
    return if mse <= @error_threshold
  end
end
