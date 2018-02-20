class KNearestNeighbors
  attr_reader :x_train, :y_train

  def fit(x, y)
    @x_train, @y_train = x, y
  end

  def predict(x_inputs)
    x_inputs.map { |x| closest(x) }
  end

  def closest(feature)
    nearest_dist = euclidian_distance(feature, @x_train[0])
    nearest_idx  = 0

    @x_train.each_with_index do |x, idx|
      dist = euclidian_distance(feature, x)
      if dist < nearest_dist
        nearest_dist = dist
        nearest_idx  = idx
      end
    end

    [@y_train[nearest_idx], feature]
  end

  def euclidian_distance(vector1, vector2)
    result = vector1.zip(vector2).reduce(0) do |sum, pair|
      sum += (pair[0] - pair[1]) ** 2
    end
    Math.sqrt(result)
  end
end

# age:       18-?
# education: 0-3
# stratum:   1-6
x_train = [
  {age: 30, education: 2, stratum: 5},
  {age: 18, education: 2, stratum: 3},
  {age: 45, education: 3, stratum: 6},
  {age: 22, education: 2, stratum: 3},
  {age: 40, education: 1, stratum: 3},
  {age: 60, education: 0, stratum: 2},
  {age: 85, education: 1, stratum: 1},
  {age: 85, education: 1, stratum: 1},
]
y_train = ["Mockus", "Mockus", "Mockus", "Mockus", "Uribe", "Uribe", "Uribe", "Uribe"]

# Training
knn = KNearestNeighbors.new
knn.fit(x_train.map{|x| [x[:education], x[:stratum]]}, y_train)

# Predictions
new_data = [[1, 2], [3, 3], [1,3], [2,2], [3,1], [1,2]]
predictions = knn.predict(new_data)
p predictions
