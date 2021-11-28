from benchmark_api.nasbench.api import NASBench
from benchmark_api.nasbench.lib import model_spec as _model_spec

ModelSpec = _model_spec.ModelSpec


class OutOfDomainError(Exception):
    """Indicates that the requested graph is outside of the search domain."""


class NASBench_(NASBench):
    def get_module_hash(self, model_spec):
        return self._hash_spec(model_spec)

    def query(self, model_spec, epochs=108, stop_halfway=False):
        """Fetch one of the evaluations for this model spec.

            Each call will sample one of the config['num_repeats'] evaluations of the
            model. This means that repeated queries of the same model (or isomorphic
            models) may return identical metrics.

            This function will increment the budget counters for benchmarking purposes.
            See self.training_time_spent, and self.total_epochs_spent.

            This function also allows querying the evaluation metrics at the halfway
            point of training using stop_halfway. Using this option will increment the
            budget counters only up to the halfway point.

            Args:
              model_spec: ModelSpec object.
              epochs: number of epochs trained. Must be one of the evaluated number of
                epochs, [4, 12, 36, 108] for the full dataset.
              stop_halfway: if True, returned dict will only contain the training time
                and accuracies at the halfway point of training (num_epochs/2).
                Otherwise, returns the time and accuracies at the end of training
                (num_epochs).

            Returns:
              dict containing the evaluated data for this object.

            Raises:
              OutOfDomainError: if model_spec or num_epochs is outside the search space.
            """
        fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)

        computed_stat_0 = computed_stat[epochs][0]
        computed_stat_1 = computed_stat[epochs][1]
        computed_stat_2 = computed_stat[epochs][2]

        data = dict()
        data['module_adjacency'] = fixed_stat['module_adjacency']
        data['module_operations'] = fixed_stat['module_operations']
        data['trainable_parameters'] = fixed_stat['trainable_parameters']

        if stop_halfway:
            data['training_time'] = computed_stat['halfway_training_time']
            data['train_accuracy'] = computed_stat['halfway_train_accuracy']
            data['validation_accuracy'] = computed_stat['halfway_validation_accuracy']
            data['test_accuracy'] = computed_stat['halfway_test_accuracy']
        else:
            data['training_time'] = computed_stat_0['final_training_time']
            data['training_time'] += computed_stat_1['final_training_time']
            data['training_time'] += computed_stat_2['final_training_time']
            data['training_time'] /= 3

            data['train_accuracy'] = computed_stat_0['final_train_accuracy']
            data['train_accuracy'] += computed_stat_1['final_train_accuracy']
            data['train_accuracy'] += computed_stat_2['final_train_accuracy']
            data['train_accuracy'] /= 3

            data['validation_accuracy'] = computed_stat_0['final_validation_accuracy']
            data['validation_accuracy'] += computed_stat_1['final_validation_accuracy']
            data['validation_accuracy'] += computed_stat_2['final_validation_accuracy']
            data['validation_accuracy'] /= 3

            data['test_accuracy'] = computed_stat_0['final_test_accuracy']
            data['test_accuracy'] += computed_stat_1['final_test_accuracy']
            data['test_accuracy'] += computed_stat_2['final_test_accuracy']
            data['test_accuracy'] /= 3

        self.training_time_spent += data['training_time']
        if stop_halfway:
            self.total_epochs_spent += epochs // 2
        else:
            self.total_epochs_spent += epochs

        return data
