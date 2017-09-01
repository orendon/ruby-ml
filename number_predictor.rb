require 'bigdecimal'
require_relative 'mlp'

network = MLP.new()
network.add_layer(10, 20) # hidden layer
network.add_layer(2, 10)  # output layer

inputs = [1, 1, 1, 1,
          1, 0, 0, 1,
          0, 0, 1, 0,
          0, 1, 0, 0,
          1, 1, 1, 1];

# output == [1, 0]

output = network.predict(inputs);
number = output.map{ |o| BigDecimal(o.to_s).round }.join

printf "Seems to be a %s %s\n", number.to_i(2), output
