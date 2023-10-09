import unittest
from model_training import train_model
from front_end import get_random_of_class, CLASSES

class TestMethods(unittest.TestCase):
    
    """ For each class instance, assert the method used by the front end to query a random class works """
    def test_random_of_class(self):
        for c in CLASSES:
            result = get_random_of_class(c)
            assert result
            
    """ Attempt to train the model using a default set of parameters """
    def test_train_model(self):
        # run one instance of training to verify that works successfully
        result = train_model(percent_of_data=0.1, num_epochs=5, batch_size=128, learning_rate=0.01)
        assert result