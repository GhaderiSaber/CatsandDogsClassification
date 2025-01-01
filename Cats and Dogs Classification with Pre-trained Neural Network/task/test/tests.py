import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy

from hstest import StageTest, CheckResult, dynamic_test
import pickle


def test_labels():
    return numpy.array([0] * 25 + [1] * 25)


class UnfreezeTest(StageTest):

    @dynamic_test()
    def test1(self):
        if not os.path.exists('../SavedModels'):
            return CheckResult.wrong(
                '`SavedModels` directory does not exists. Make sure you create and run your solution before testing!')

        if not os.path.exists('../SavedHistory'):
            return CheckResult.wrong(
                '`SavedModels` directory does not exists. Make sure you create and run your solution before testing!')

        return CheckResult.correct()

    @dynamic_test(time_limit=60000)
    def test2(self):

        if 'stage_five_history' not in os.listdir('../SavedHistory'):
            return CheckResult.wrong("The file `stage_five_history` is not in SavedHistory directory")

        with open('../SavedHistory/stage_five_history', 'rb') as stage_five:
            answer = pickle.load(stage_five)

        if not isinstance(answer, numpy.ndarray):
            return CheckResult.wrong("`stage_five_history` should be a numpy array")

        labels = test_labels()
        accuracy = labels == answer

        if labels.shape != answer.shape:
            return CheckResult.wrong(
                f"Shape of the `stage_five_history` is wrong. It must be {labels.shape}, not {answer.shape}!")

        test_accuracy = accuracy.mean()

        if test_accuracy < 0.95:
            return CheckResult.wrong(f"Your model's accuracy is {test_accuracy * 100}%\n"
                                     "The goal is to score at least 95%")

        print(f"Test accuracy: {round(test_accuracy, 3)}")
        return CheckResult.correct()


if __name__ == '__main__':
    UnfreezeTest().run_tests()