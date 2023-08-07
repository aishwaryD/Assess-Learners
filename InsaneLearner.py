import BagLearner as bl
import LinRegLearner as ll

class InsaneLearner(object):

    def __init__(self, verbose=False): self.learners = [bl.BagLearner(learner=ll.LinRegLearner, kwargs={}, bags=20, boost=False, verbose=False) for _ in range(20)]

    def author(self):
        return 'aishwaryD'

    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)

    def query(self, points):
        return sum(learner.query(points) for learner in self.learners) / len(self.learners)

if __name__ == "__main__":
    print("the secret clue is 'adwivedi62'")
