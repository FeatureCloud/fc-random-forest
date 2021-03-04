import os
import pickle
import threading
import time
import yaml

import pandas as pd

from distutils import dir_util

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


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
        self.train = None
        self.sep = None
        self.label_column = None
        self.data = None
        self.data_X = None
        self.data_y = None
        self.data_X_train = None
        self.data_y_train = None
        self.data_X_test = None
        self.data_y_test = None

        self.filename = None
        self.values = None

        self.rf = None

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
        print(f'Received data: {data}')
        # This method is called when new data arrives
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        print(f'Submit data: {self.data_outgoing}')
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def read_config(self):
        dir_util.copy_tree('/mnt/input/', '/mnt/output/')
        with open('/mnt/input/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_random_forest']
            self.train = config['files']['input']
            self.sep = config['files']['sep']
            self.label_column = config['files']['label_column']
            self.estimators = config['params'].get('estimators', 100)
            self.test_split = config['test'].get('split', 0.3)
            self.mode = config.get('mode', 'classification')

    def app_flow(self):
        # This method contains a state machine for the slave and master instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_train_local = 2
        state_gather = 3
        state_wait = 4
        state_global_ready = 5
        state_finishing = 6

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
                self.data = pd.read_csv(os.path.join(f'/mnt/input/', self.train), sep=self.sep)
                self.data_X = self.data.drop(self.label_column, axis=1)
                self.data_y = self.data[self.label_column]
                self.data_X_train, self.data_X_test, self.data_y_train, self.data_y_test = train_test_split(self.data_X, self.data_y, test_size=self.test_split)
                print('Read input.')
                state = state_train_local

            if state == state_train_local:
                print('Calculate local values...')

                rf = None
                if self.mode == 'classification':
                    rf = RandomForestClassifier(n_estimators=self.estimators)
                elif self.mode == 'regression':
                    rf = RandomForestRegressor(n_estimators=self.estimators)
                rf.fit(self.data_X_train, self.data_y_train)

                print(f'Trained random forest')

                if self.master:
                    self.data_incoming.append(pickle.dumps({
                        'rf': rf,
                    }))
                    state = state_gather
                else:
                    self.data_outgoing = pickle.dumps({
                        'rf': rf,
                    })
                    self.status_available = True
                    state = state_wait

            if state == state_global_ready:
                print(f'Forest done')

                if self.mode == 'classification':
                    y_proba = self.rf.predict_proba(self.data_X_test)

                    y_proba_df = pd.DataFrame(data={'prob_0': y_proba[:, 0], 'prob_1': y_proba[:, 1]})
                    y_proba_df.to_csv(os.path.join('/mnt/output/', 'y_proba.csv'), index=False, sep=self.sep)
                elif self.mode == 'regression':
                    y_pred = self.rf.predict(self.data_X_test)

                    y_pred_df = pd.DataFrame(data={'pred': y_pred[:]})
                    y_pred_df.to_csv(os.path.join('/mnt/output/', 'y_pred.csv'), index=False, sep=self.sep)

                y_true_df = pd.DataFrame(data={'y_true': self.data_y_test})
                y_true_df.to_csv(os.path.join('/mnt/output/', 'y_true.csv'), index=False, sep=self.sep)

                if self.master:
                    self.data_incoming = ['DONE']
                    state = state_finishing
                else:
                    self.data_outgoing = 'DONE'
                    self.status_available = True
                    break

            # GLOBAL PART

            if state == state_gather:
                if len(self.data_incoming) == len(self.clients):
                    rf = None
                    for d in self.data_incoming:
                        d = pickle.loads(d)
                        drf = d['rf']

                        if rf is None:
                            rf = drf
                        else:
                            rf.estimators_ += drf.estimators_
                            rf.n_estimators += drf.n_estimators

                    self.rf = rf

                    self.data_outgoing = pickle.dumps({
                        'rf': rf,
                    })
                    self.status_available = True
                    state = state_global_ready

                else:
                    print(f'Have {len(self.data_incoming)} of {len(self.clients)} so far, waiting...')

            if state == state_finishing:
                if len(self.data_incoming) == len(self.clients):
                    self.status_finished = True
                    break

            # LOCAL PART

            if state == state_wait:
                if len(self.data_incoming) > 0:
                    d = pickle.loads(self.data_incoming[0])
                    self.rf = d['rf']
                    state = state_global_ready

            time.sleep(1)


logic = AppLogic()
