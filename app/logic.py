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


APP_NAME = 'fc_random_forest'


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for master, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.master = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.input_train = None
        self.input_test = None
        self.output_pred = None
        self.output_proba = None
        self.output_test = None
        self.split_mode = None
        self.split_dir = None
        self.split_test = None

        self.sep = None
        self.label = None

        self.my_samples = None
        self.total_samples = None

        self.estimators_total = None
        self.mode = None
        self.random_state = None

        self.data_X_train = []
        self.data_y_train = []
        self.data_X_test = []
        self.data_y_test = []

        self.values = None

        self.rfs = None

        self.lock = threading.Lock()

    def handle_setup(self, client_id, master, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.master = master
        self.clients = clients
        print(f'Received setup: {self.id} {self.master} {self.clients}')

        self.read_config()

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        print(f'Received data')
        # This method is called when new data arrives
        self.lock.acquire()
        self.data_incoming.append(data.read())
        self.lock.release()

    def handle_outgoing(self):
        print(f'Submit data')
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def read_config(self):
        dir_util.copy_tree('/mnt/input/', '/mnt/output/')
        with open('/mnt/input/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)[APP_NAME]

            self.input_train = config['input'].get('train')
            self.input_test = config['input'].get('test')

            self.output_pred = config['output'].get('pred')
            self.output_proba = config['output'].get('proba')
            self.output_test = config['output'].get('test')

            self.split_mode = config['split'].get('mode')
            self.split_dir = config['split'].get('dir')
            self.split_test = config['split'].get('test')

            self.sep = config['format']['sep']
            self.label = config['format']['label']

            self.estimators_total = config.get('estimators', 100)
            self.mode = config.get('mode', 'classification')
            self.random_state = config.get('random_state')

    def app_flow(self):
        # This method contains a state machine for the slave and master instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_share_samples = 3
        state_gather_1 = 4
        state_wait_1 = 5
        state_train_local = 6
        state_gather_2 = 7
        state_wait_2 = 8
        state_global_ready = 9
        state_finishing = 10

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:

            if state == state_initializing:
                if self.id is not None:  # Test if setup has happened already
                    state = state_read_input

            # COMMON PART

            if state == state_read_input:
                print('Reading input...')
                base_dir = os.path.normpath(os.path.join(f'/mnt/input/', self.split_dir))

                def read_input_train(ins, path):
                    d = pd.read_csv(path, sep=self.sep)
                    data_X = d.drop(self.label, axis=1)
                    data_y = d[self.label]

                    if ins.split_test is not None:
                        ins.data = pd.read_csv(os.path.join(base_dir, ins.input_train), sep=ins.sep)
                        data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_X, data_y, test_size=ins.split_test)
                        ins.data_X_train.append(data_X_train)
                        ins.data_y_train.append(data_y_train)
                        ins.data_X_test.append(data_X_test)
                        ins.data_y_test.append(data_y_test)
                    else:
                        ins.data_X_train.append(data_X)
                        ins.data_y_train.append(data_y)

                def read_input_test(ins, path):
                    d = pd.read_csv(path, sep=ins.sep)
                    data_X = d.drop(ins.label, axis=1)
                    data_y = d[ins.label]
                    ins.data_X_test.append(data_X)
                    ins.data_y_test.append(data_y)

                if self.split_mode == 'directory':
                    for split_name in os.listdir(base_dir):
                        read_input_train(self, os.path.join(base_dir, split_name, self.input_train))
                        if self.input_test is not None:
                            read_input_test(self, os.path.join(base_dir, split_name, self.input_test))
                elif self.split_mode == 'file':
                    read_input_train(self, os.path.join(base_dir, self.input_train))
                    if self.input_test is not None:
                        read_input_test(self, os.path.join(base_dir, self.input_test))

                split_samples = [i.shape[0] for i in self.data_y_train]
                self.my_samples = sum(split_samples) // len(split_samples)

                print(f'Read input. Have {split_samples} samples.')

                if self.master:
                    self.data_incoming.append(pickle.dumps({
                        'samples': self.my_samples
                    }))
                    state = state_gather_1
                else:
                    self.data_outgoing = pickle.dumps({
                        'samples': self.my_samples
                    })
                    self.status_available = True
                    state = state_wait_1

            if state == state_train_local:
                print('Calculate local values...')

                rfs = []
                for i in range(len(self.data_X_train)):
                    global_rf = None
                    trees = int(self.estimators_total * self.my_samples / self.total_samples)
                    if self.mode == 'classification':
                        global_rf = RandomForestClassifier(n_estimators=trees, random_state=self.random_state)
                    elif self.mode == 'regression':
                        global_rf = RandomForestRegressor(n_estimators=trees, random_state=self.random_state)
                    global_rf.fit(self.data_X_train[i], self.data_y_train[i])
                    rfs.append({
                        'rf': global_rf,
                    })

                print(f'Trained random forests')

                if self.master:
                    self.data_incoming.append(pickle.dumps(rfs))
                    state = state_gather_2
                else:
                    self.data_outgoing = pickle.dumps(rfs)
                    self.status_available = True
                    state = state_wait_2

            if state == state_global_ready:
                print(f'Forest done')

                results_pred = []
                results_proba = []
                results_test = []
                for i in range(len(self.data_X_train)):
                    results_pred.append(self.rfs[i].predict(self.data_X_test[i]))
                    if self.mode == 'classification':
                        results_proba.append(self.rfs[i].predict_proba(self.data_X_test[i]))
                    results_test.append(self.data_y_test[i])

                def write_output(path, data):
                    df = pd.DataFrame(data=data)
                    df.to_csv(path, index=False, sep=self.sep)

                print(f'Writing output')
                base_dir_in = os.path.normpath(os.path.join(f'/mnt/input/', self.split_dir))
                base_dir_out = os.path.normpath(os.path.join(f'/mnt/output/', self.split_dir))
                if self.split_mode == 'directory':
                    for i, split_name in enumerate(os.listdir(base_dir_in)):
                        write_output(os.path.join(base_dir_out, split_name, self.output_pred), {'pred': results_pred[i][:]})
                        if self.mode == 'classification':
                            write_output(os.path.join(base_dir_out, split_name, self.output_proba), {'prob_0': results_proba[i][:, 0], 'prob_1': results_proba[i][:, 1]})
                        write_output(os.path.join(base_dir_out, split_name, self.output_test), {'y_true': results_test[i]})
                elif self.split_mode == 'file':
                    write_output(os.path.join(base_dir_out, self.output_pred), {'pred': results_pred[0][:]})
                    if self.mode == 'classification':
                        write_output(os.path.join(base_dir_out, self.output_proba), {'prob_0': results_proba[0][:, 0], 'prob_1': results_proba[0][:, 1]})
                    write_output(os.path.join(base_dir_out, self.output_test), {'y_true': results_test[0]})

                if self.master:
                    self.data_incoming.append('DONE')
                    state = state_finishing
                else:
                    self.data_outgoing = 'DONE'
                    self.status_available = True
                    break

            # GLOBAL PART

            if state == state_gather_1:
                if len(self.data_incoming) == len(self.clients):

                    client_data = []
                    for local_rfs in self.data_incoming:
                        client_data.append(pickle.loads(local_rfs))

                    self.data_incoming = []

                    total_samples = sum([cd['samples'] for cd in client_data])

                    self.total_samples = total_samples

                    self.data_outgoing = pickle.dumps(total_samples)
                    self.status_available = True
                    state = state_train_local

                else:
                    print(f'Have {len(self.data_incoming)} of {len(self.clients)} so far, waiting...')

            if state == state_gather_2:
                if len(self.data_incoming) == len(self.clients):

                    client_data = []
                    for local_rfs in self.data_incoming:
                        client_data.append(pickle.loads(local_rfs))

                    self.data_incoming = []

                    data_outgoing = []

                    for i in range(len(self.data_X_train)):
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

                    self.rfs = data_outgoing

                    self.data_outgoing = pickle.dumps(data_outgoing)
                    self.status_available = True
                    state = state_global_ready

                else:
                    print(f'Have {len(self.data_incoming)} of {len(self.clients)} so far, waiting...')

            if state == state_finishing:
                if len(self.data_incoming) == len(self.clients):
                    self.status_finished = True
                    break

            # LOCAL PART

            if state == state_wait_1:
                if len(self.data_incoming) > 0:
                    self.total_samples = pickle.loads(self.data_incoming[0])
                    self.data_incoming = []

                    state = state_train_local

            if state == state_wait_2:
                if len(self.data_incoming) > 0:
                    self.rfs = pickle.loads(self.data_incoming[0])
                    self.data_incoming = []

                    state = state_global_ready

            time.sleep(1)


logic = AppLogic()
