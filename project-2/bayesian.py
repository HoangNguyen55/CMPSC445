class BayesianClassifier:
    def __init__(self):
        # Step 2, feature dataset (bag of words) for each category
        self.word_counts = {'positive': {}, 'negative': {}}
        self.class_counts = {'positive': 0, 'negative': 0}
        self.vocab = set()

    # step 4, training
    def train(self, training_data):
        for data_point in training_data:
            review, category = data_point
            words = review.split(' ')
            for word in words:
                self.word_counts[category][word] = self.word_counts[category].get(word, 0) + 1
                self.vocab.add(word)
            self.class_counts[category] += 1

    def calc_prior(self, category):
        total = sum(self.class_counts.values())
        return self.class_counts[category] / total

    def calc_cond_prob(self, word, category):
        word_count = self.word_counts[category].get(word, 0)
        total_words_in_category = sum(self.word_counts[category].values())
        # laplace smoothing, https://en.wikipedia.org/wiki/Additive_smoothing
        return (word_count + 1) / (total_words_in_category + len(self.vocab))

    def classify(self, text):
        words = text.split(' ')
        # step 5, calc prior prob and conditional prob
        positive_score = self.calc_prior('positive')
        negative_score = self.calc_prior('negative')

        for word in words:
            if word in self.vocab:
                # P(yes) * P(word|yes) * ...
                positive_score *= self.calc_cond_prob(word, 'positive')
                # P(no) * P(word|no) * ...
                negative_score *= self.calc_cond_prob(word, 'negative')

        return 'positive' if positive_score > negative_score else 'negative'


def data_preprocessing(file):
    data = []
    with open(f"{file}.txt") as p:
        strings = p.readlines()
        for string in strings:
            data.append((string, file))

    return data

def split_data(data, train_percentage):
    from random import shuffle 
    # randomize the list
    shuffle(data)
    ceil = int(len(data) * train_percentage/100)
    return (data[:ceil], data[ceil:])

def run_bayesian(training_data, test_data):
    classifier = BayesianClassifier()
    classifier.train(training_data)

    # Step 6, classify testing data
    review_correct = []
    for review in test_data:
        category = classifier.classify(review[0])
        # Step 7/9, record result
        review_correct.append(int(category == review[1]))
        # print(f"Predicted correctly? ", category == review[1])
    print(f"Correct ratio {sum(review_correct)}/{len(review_correct)}, {sum(review_correct)/len(review_correct)*100}%")

if __name__ == "__main__":
    # Step 1, data collected from https://ai.stanford.edu/~amaas/data/sentiment/
    # only take the 1st 50 review from postive and negative for 100 reviews in total
    data = data_preprocessing("positive")
    data += data_preprocessing("negative")

    # Step 3, seperate testing/training data
    training_data, test_data = split_data(data, 50)
    run_bayesian(training_data, test_data)

    # Step 8, 80/20 split
    training_data, test_data = split_data(data, 80)
    run_bayesian(training_data, test_data)
