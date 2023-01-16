import os
import pickle
import random
import threading
import time
import yaml

import pandas as pd

from distutils import dir_util

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel, State

@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    Initialize client.
    """
    
    def register(self):
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        print("Initializing")
        if self.id is not None:  # Test if setup has happened already
            print("Coordinator", self.is_coordinator)
        self.init()
        return 'read input'
        
    def init(self):
        self.store('input_train', None)
        self.store('input_test', None)
        self.store('output_pred', None)
        self.store('output_proba', None)
        self.store('output_test', None)
        self.store('split_mode', None)
        self.store('split_dir', None)
        self.store('split_test', None)

        self.store('sep', None)
        self.store('label', None)

        self.store('my_samples', None)
        self.store('total_samples', None)

        self.store('estimators_total', None)
        self.store('mode', None)
        self.store('random_state', None)

        self.store('values', None)

        self.store('rfs', None)
       
       
# COMMON PART

@app_state('read input', Role.BOTH)
class ReadInputState(AppState):
    """
    Read input data and config file.
    """
    
    def register(self):
        self.register_transition('gather1', Role.COORDINATOR)
        self.register_transition('wait1', Role.PARTICIPANT)
        self.register_transition('read input', Role.BOTH)

    def run(self) -> str or None:
        try:
            print('Reading input...')
            self.read_config()
            base_dir = os.path.normpath(os.path.join(f'/mnt/input/', self.load('split_dir')))

            def read_input_train(path):
                d = pd.read_csv(path, sep=self.load('sep'))
                data_X = d.drop(self.load('label'), axis=1)
                data_y = d[self.load('label')]

                if self.load('split_test') is not None:
                    data = pd.read_csv(os.path.join(base_dir, ins.input_train), sep=self.load('sep'))
                    return train_test_split(data_X, data_y, test_size=self.load('split_test'))
                   
                else:
                    return data_X, data_y

            def read_input_test(path):
                d = pd.read_csv(path, sep=self.load('sep'))
                data_X = d.drop(self.load('label'), axis=1)
                data_y = d[self.load('label')]
                return data_X, data_y

            data_X_train = []
            data_y_train = []
            data_X_test = []
            data_y_test = []

            if self.load('split_mode') == 'directory':
                for split_name in os.listdir(base_dir):
                    if self.load('split_test') is not None:
                        data_X_train, data_X_test, data_y_train, data_y_test = read_input_train(os.path.join(base_dir, split_name, self.load('input_train')))
                        data_X_train.append(data_X_train)
                        data_y_train.append(data_y_train)
                        data_X_test.append(data_X_test)
                        data_y_test.append(data_y_test)
                    else:
                        data_X, data_y = read_input_train(os.path.join(base_dir, split_name, self.load('input_train')))
                        data_X_train.append(data_X)
                        data_y_train.append(data_y)
                        
                    if self.load('input_test') is not None:
                        data_X, data_y = read_input_test(os.path.join(base_dir, split_name, self.load('input_test')))
                        data_X_test.append(data_X)
                        data_y_test.append(data_y)
                        
            elif self.load('split_mode') == 'file':
                if self.load('split_test') is not None:
                    data_X_train, data_X_test, data_y_train, data_y_test = read_input_train(os.path.join(base_dir, self.load('input_train')))
                    data_X_train.append(data_X_train)
                    data_y_train.append(data_y_train)
                    data_X_test.append(data_X_test)
                    data_y_test.append(data_y_test)
                else:
                    data_X, data_y = read_input_train(os.path.join(base_dir, self.load('input_train')))
                    data_X_train.append(data_X)
                    data_y_train.append(data_y)
                    
                if self.load('input_test') is not None:
                    data_X, data_y = read_input_test(os.path.join(base_dir, self.load('input_test')))
                    data_X_test.append(data_X)
                    data_y_test.append(data_y)
            
            self.store('data_X_train', data_X_train)
            self.store('data_y_train', data_y_train)
            self.store('data_X_test', data_X_test)
            self.store('data_y_test', data_y_test)
            
            split_samples = [i.shape[0] for i in self.load('data_y_train')]
            self.store('my_samples', sum(split_samples) // len(split_samples))

            print(f'Read input. Have {split_samples} samples.')

            data_to_send = pickle.dumps({'samples': self.load('my_samples')})
            self.send_data_to_coordinator(data_to_send)
            if self.is_coordinator:
                return 'gather1'
            else:
                return 'wait1'

        except Exception as e:
            self.log('no config file or missing fields', LogLevel.ERROR)
            self.update(message='no config file or missing fields', state=State.ERROR)
            print(e)
            return 'read input'

    def read_config(self):
        dir_util.copy_tree('/mnt/input/', '/mnt/output/')
        with open('/mnt/input/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_random_forest']

            self.store('input_train', config['input'].get('train'))
            self.store('input_test', config['input'].get('test'))

            self.store('output_pred', config['output'].get('pred'))
            self.store('output_proba', config['output'].get('proba'))
            self.store('output_test', config['output'].get('test'))

            self.store('split_mode', config['split'].get('mode'))
            self.store('split_dir', config['split'].get('dir'))
            self.store('split_test', config['split'].get('test'))

            self.store('sep', config['format']['sep'])
            self.store('label', config['format']['label'])

            self.store('estimators_total', config.get('estimators', 100))
            self.store('mode', config.get('mode', 'classification'))
            self.store('random_state', config.get('random_state'))
            
            
@app_state('train local', Role.BOTH)
class TrainLocalState(AppState):
    """
    Perform the local training.
    """
    
    def register(self):
        self.register_transition('gather2', Role.COORDINATOR)
        self.register_transition('wait2', Role.PARTICIPANT)
        self.register_transition('train local', Role.BOTH)

    def run(self) -> str or None:
        try:
            print('Calculate local values...')

            rfs = []
            for i in range(len(self.load('data_X_train'))):
                global_rf = None
                trees = int(self.load('estimators_total') * self.load('my_samples') / self.load('total_samples'))
                if self.load('mode') == 'classification':
                    global_rf = RandomForestClassifier(n_estimators=trees, random_state=self.load('random_state'))
                elif self.load('mode') == 'regression':
                    global_rf = RandomForestRegressor(n_estimators=trees, random_state=self.load('random_state'))
                global_rf.fit(self.load('data_X_train')[i], self.load('data_y_train')[i])
                rfs.append({'rf': global_rf,})

            print(f'Trained random forests')
            self.send_data_to_coordinator(pickle.dumps(rfs))
            if self.is_coordinator:
                return 'gather2'
            else:
                return 'wait2'
            
        except Exception as e:
            self.log('error train local', LogLevel.ERROR)
            self.update(message='error train local', state=State.ERROR)
            print(e)
            return 'train local'


@app_state('global ready', Role.BOTH)
class GlobalReadyState(AppState):
    """
    Writes the results.
    """
    
    def register(self):
        self.register_transition('finishing', Role.COORDINATOR)
        self.register_transition('terminal', Role.PARTICIPANT)
        self.register_transition('global ready', Role.BOTH)

    def run(self) -> str or None:
        try:
            print(f'Forest done')

            results_pred = []
            results_proba = []
            results_test = []
            rfs = self.load('rfs')
            for i in range(len(self.load('data_X_train'))):
                results_pred.append(rfs[i].predict(self.load('data_X_test')[i]))
                if self.load('mode') == 'classification':
                    results_proba.append(rfs[i].predict_proba(self.load('data_X_test')[i]))
                results_test.append(self.load('data_y_test')[i])

            def write_output(path, data):
                df = pd.DataFrame(data=data)
                df.to_csv(path, index=False, sep=self.load('sep'))

            print(f'Writing output')
            base_dir_in = os.path.normpath(os.path.join(f'/mnt/input/', self.load('split_dir')))
            base_dir_out = os.path.normpath(os.path.join(f'/mnt/output/', self.load('split_dir')))
            if self.load('split_mode') == 'directory':
                for i, split_name in enumerate(os.listdir(base_dir_in)):
                    write_output(os.path.join(base_dir_out, split_name, self.load('output_pred')), {'pred': results_pred[i][:]})
                    if self.load('mode') == 'classification':
                        write_output(os.path.join(base_dir_out, split_name, self.load('output_proba')), {'prob_0': results_proba[i][:, 0], 'prob_1': results_proba[i][:, 1]})
                    write_output(os.path.join(base_dir_out, split_name, self.load('output_test')), {'y_true': results_test[i]})
            elif self.load('split_mode') == 'file':
                write_output(os.path.join(base_dir_out, self.load('output_pred')), {'pred': results_pred[0][:]})
                if self.load('mode') == 'classification':
                    write_output(os.path.join(base_dir_out, self.load('output_proba')), {'prob_0': results_proba[0][:, 0], 'prob_1': results_proba[0][:, 1]})
                write_output(os.path.join(base_dir_out, self.load('output_test')), {'y_true': results_test[0]})
            
            self.send_data_to_coordinator('DONE')
            
            if self.is_coordinator:
                return 'finishing'
            else:
                return 'terminal'

        except Exception as e:
            self.log('error global ready', LogLevel.ERROR)
            self.update(message='error global ready', state=State.ERROR)
            print(e)
            return 'global ready'

# GLOBAL PART
@app_state('gather1', Role.COORDINATOR)
class Gather1State(AppState):
    """
    The coordinator receives the local trained data from each client and aggregates the data.
    The coordinator broadcasts the global data to the clients.
    """
    
    def register(self):
        self.register_transition('train local', Role.COORDINATOR)
        self.register_transition('gather1', Role.COORDINATOR)

    def run(self) -> str or None:
        try:
            data = self.gather_data()
            client_data = []

            for local_rfs in data:
                client_data.append(pickle.loads(local_rfs))

            total_samples = sum([cd['samples'] for cd in client_data])

            self.store('total_samples', total_samples)
            data_to_broadcast = pickle.dumps(total_samples)
            self.broadcast_data(data_to_broadcast, send_to_self=False)
            return 'train local'

        except Exception as e:
            self.log('error gather1', LogLevel.ERROR)
            self.update(message='error gather1', state=State.ERROR)
            print(e)
            return 'gather1'


@app_state('gather2', Role.COORDINATOR)
class Gather2State(AppState):
    """
    The coordinator receives the local trained data from each client and aggregates the data.
    The coordinator broadcasts the global data to the clients.
    """
    
    def register(self):
        self.register_transition('global ready', Role.COORDINATOR)
        self.register_transition('gather2', Role.COORDINATOR)

    def run(self) -> str or None:
        try:
            data = self.gather_data()
            client_data = []
            for local_rfs in data:
                client_data.append(pickle.loads(local_rfs))

                data_outgoing = []

                for i in range(len(self.load('data_X_train'))):
                    global_rf = None

                    # total_samples = 0
                    # for d in client_data:
                    #     total_samples += d[i]['samples']

                    for d in client_data:
                        drf = d[i]['rf']

                        # perc = d[i]['samples'] / total_samples
                        # trees = int(perc * self.estimators_total)

                        if global_rf is None:
                            global_rf = drf
                            global_rf.estimators_ = drf.estimators_
                            # global_rf.estimators_ = random.sample(drf.estimators_, trees)
                            global_rf.n_estimators = drf.n_estimators
                        else:
                            global_rf.estimators_ += drf.estimators_
                            # global_rf.estimators_ += random.sample(drf.estimators_, trees)
                            global_rf.n_estimators += drf.n_estimators

                    data_outgoing.append(global_rf)

                self.store('rfs', data_outgoing)

                data_to_broadcast = pickle.dumps(data_outgoing)
                self.broadcast_data(data_to_broadcast, send_to_self=False)
                
                return 'global ready'

        except Exception as e:
            self.log('error gather2', LogLevel.ERROR)
            self.update(message='error gather2', state=State.ERROR)
            print(e)
            return 'gather2'


@app_state('finishing', Role.COORDINATOR)
class FinishingState(AppState):
    
    def register(self):
        self.register_transition('terminal', Role.COORDINATOR)
        self.register_transition('finishing', Role.COORDINATOR)

    def run(self) -> str or None:
        try:
            self.gather_data()
            return 'terminal'

        except Exception as e:
            self.log('error finishing', LogLevel.ERROR)
            self.update(message='error finishing', state=State.ERROR)
            print(e)
            return 'finishing'

# LOCAL PART
@app_state('wait1', Role.PARTICIPANT)
class Wait1State(AppState):
    """
    Wait for the aggregation result.
    """
    
    def register(self):
        self.register_transition('train local', Role.PARTICIPANT)
        self.register_transition('wait1', Role.PARTICIPANT)

    def run(self) -> str or None:
        try:
            data = self.await_data()
            self.store('total_samples', pickle.loads(data))
            return 'train local'

        except Exception as e:
            self.log('error wait1', LogLevel.ERROR)
            self.update(message='error wait1', state=State.ERROR)
            print(e)
            return 'wait1'


@app_state('wait2', Role.PARTICIPANT)
class Wait2State(AppState):
    """
    Wait for the aggregation result.
    """
    
    def register(self):
        self.register_transition('global ready', Role.PARTICIPANT)
        self.register_transition('wait2', Role.PARTICIPANT)

    def run(self) -> str or None:
        try:
            data = self.await_data()
            self.store('rfs', pickle.loads(data))
            return 'global ready'

        except Exception as e:
            self.log('error wait2', LogLevel.ERROR)
            self.update(message='error wait2', state=State.ERROR)
            print(e)
            return 'wait2'
