module MathUtils
  def self.random
    randomizer = Random.new
    randomizer.rand(-0.2..0.2)
  end

  def self.mse(errors)
    errors.reduce(0) do |sum, err|
      sum + err ** 2
    end
  end

  def self.sigmoid(z)
    1 / (1 + Math.exp(-z))
  end

  def self.sigmoid_derivative(x)
    sigmoid(x) * (1 - sigmoid(x))
  end
end
