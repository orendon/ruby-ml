require 'bigdecimal'
require_relative 'mlp'
require_relative 'training_data'

network = MLP.new()
network.add_layer(10, 20) # hidden layer
network.add_layer(2, 10)  # output layer


# Training / Learning

network.learn([
                [zero,  [0, 0]],
                [one,   [0, 1]],
                [two,   [1, 0]],
                [three, [1, 1]]
              ])


# Actual Prediction

inputs = [1, 1, 1, 1,
          1, 0, 0, 1,
          0, 0, 1, 0,
          0, 1, 0, 0,
          1, 1, 1, 1]

output = network.predict(inputs)
number = output.map{ |o| BigDecimal(o, 10).round }.join

printf "Seems to be a %s %s\n", number.to_i(2), output
