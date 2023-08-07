import numpy as np


class BagLearner(object):

    def __init__(self, learner, bags, kwargs=None, boost=False, verbose=False):
        self.boost = boost
        self.verbose = verbose
        self.learners = [learner(**kwargs) for _ in range(bags)]

    def author(self):
        return 'aishwary'

    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            bag_indices = np.random.choice(range(data_x.shape[0]), size=data_x.shape[0], replace=True)
            bag_x = data_x[bag_indices]
            bag_y = data_y[bag_indices]
            learner.add_evidence(bag_x, bag_y)

    def query(self, points):
        return sum(learner.query(points) for learner in self.learners) / len(self.learners)


if __name__ == "__main__":
    print("the secret clue is 'adwivedi62'")
