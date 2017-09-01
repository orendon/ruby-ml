module MathUtils
  def self.random
    randomizer = Random.new
    randomizer.rand(-0.4..0.4)
  end

  def self.mse(errors)
    errors_sum = errors.reduce(0) do |sum, err|
      sum + err * err
    end

    errors_sum / errors.size
  end

  def self.sigmoid(z)
    1 / (1 + Math.exp(-z))
  end

  def self.sigmoid_derivative(x)
    sigmoid(x) * (1 - sigmoid(x))
  end
end
