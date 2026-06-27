import numpy as np
from scipy.io import loadmat

class CollaborativeFiltering:
    def __init__(self, movie_file, param_file, Lambda=0):
        self.movie_file = movie_file
        self.param_file = param_file
        self.Lambda = Lambda
        self.Y = None
        self.R = None
        self.X = None
        self.Theta = None
        self.num_users = 0
        self.num_movies = 0
        self.num_features = 0

    # Load Data
    def load_data(self):
        movie_data = loadmat(self.movie_file)
        self.Y = movie_data["Y"]
        self.R = movie_data["R"]
        param_data = loadmat(self.param_file)
        self.X = param_data["X"]
        self.Theta = param_data["Theta"]
        self.num_users = int(param_data["num_users"].item())
        self.num_movies = int(param_data["num_movies"].item())
        self.num_features = int(param_data["num_features"].item())
        print("Data Loaded")
        print("Y :", self.Y.shape)
        print("X :", self.X.shape)
        print("Theta :", self.Theta.shape)

    # Cost Function
    def cost_function(self, params):
        X = params[:self.num_movies*self.num_features].reshape(self.num_movies, self.num_features)
        Theta = params[self.num_movies*self.num_features:].reshape(self.num_users,self.num_features)
        prediction = X.dot(Theta.T)
        error = (prediction - self.Y) * self.R
        cost = (0.5 *np.sum(error ** 2))
        reg = (self.Lambda / 2 * (np.sum(X**2) + np.sum(Theta**2)))
        cost += reg
        return cost

    # Gradient
    def gradient(self, params):
        X = params[:self.num_movies*self.num_features].reshape(self.num_movies, self.num_features)
        Theta = params[self.num_movies*self.num_features:].reshape(self.num_users,self.num_features)
        prediction = X.dot(Theta.T)
        error = (prediction - self.Y) * self.R
        X_grad = (error.dot(Theta)+self.Lambda * X)
        Theta_grad = (error.T.dot(X) + self.Lambda * Theta)
        grad = np.concatenate([X_grad.ravel(), Theta_grad.ravel()])
        return grad

    # Predict Rating
    def predict(self):
        prediction = (self.X.dot(self.Theta.T))
        return prediction

    # Recommend Movie
    def recommend(self, user_id, top_n=10):
        prediction = self.predict()
        # selected user rating
        user_rating = prediction[:, user_id]
        # high rating first
        sorted_movie = np.argsort(user_rating)[::-1]
        print(f"\n===== User {user_id} Recommendation =====\n")
        for movie in sorted_movie[:top_n]:
            print(f"Movie {movie} "
                f"→ {user_rating[movie]:.2f}")

    # Gradient Check
    def gradient_check(self):
        print("\nChecking Gradient...")
        # fake data
        X = np.random.rand(4,3)
        Theta = np.random.rand(5,3)
        Y = X.dot(Theta.T)
        R = np.ones(Y.shape)
        movies, users = Y.shape
        features = 3
        params = np.concatenate([X.ravel(), Theta.ravel()])
        epsilon = 1e-4
        numerical = np.zeros(params.shape)
        for i in range(params.size):
            temp = params[i]
            params[i] = temp + epsilon
            loss1 = self._test_cost(params, Y, R, users, movies, features)
            params[i] = temp - epsilon
            loss2 = self._test_cost(params, Y, R, users, movies, features)
            numerical[i] = (loss1-loss2)/(2*epsilon)
            params[i] = temp
        analytical = self._test_gradient(params, Y, R, users, movies, features)
        diff = (np.linalg.norm(numerical-analytical) / np.linalg.norm(numerical+analytical))
        print("Difference:",diff)
        if diff < 1e-9:
            print("Gradient Check Passed")
        else:
            print("Gradient Check Failed")

    # test cost
    def _test_cost(self, params, Y, R, users, movies, features):
        X = params[:movies*features].reshape(movies, features)
        Theta = params[movies*features:].reshape(users, features)
        error = (X.dot(Theta.T)-Y)*R
        return 0.5*np.sum(error**2)

    # test gradient
    def _test_gradient(self, params, Y, R, users, movies, features):
        X = params[:movies*features].reshape(movies,features)
        Theta = params[movies*features:].reshape(users, features)
        error = (X.dot(Theta.T)-Y)*R
        X_grad = error.dot(Theta)
        Theta_grad = error.T.dot(X)
        return np.concatenate([X_grad.ravel(), Theta_grad.ravel()])


# Main
if __name__ == "__main__":
    movie_path = ("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex8/ex8_movies.mat")
    param_path = ("C:/Users/User/Desktop/Machine learning exercise/data1/data/ex8/ex8_movieParams.mat")
    model = CollaborativeFiltering(movie_path, param_path, Lambda=1)
    model.load_data()
    # Recommendation
    model.recommend(user_id=10,top_n=5)
    # Gradient test
    model.gradient_check()