import unittest
import matplotlib.pyplot as plt

from hpbandster.core.result import Run, logged_results_to_HBS_result
from cave.utils.hpbandster_helpers import _compute_trajectory_racing, get_incumbent_trajectory

class TestHpbandsterHelpers(unittest.TestCase):

    def setUp(self):
        self.plot = False
        self.result_path = "examples/bohb"

    def plot_incs(self, inc_dict):
        """ Simple Plot Function to test the incumbent calculation"""
        fig, (ax2, ax1) = plt.subplots(1, 2)
        plt.subplots_adjust(wspace=0.5)

        ax1.scatter(inc_dict['times_finished'], inc_dict['losses'])
        ax1.plot(inc_dict['times_finished'], inc_dict['losses'])
        ax2.scatter(inc_dict['budgets'], inc_dict['losses'])
        ax2.plot(inc_dict['budgets'], inc_dict['losses'])

        ax1.set_xlabel('finish time')
        ax2.set_xlabel('budget')

        for ax in (ax2, ax1):
            ax.set_ylabel('Loss')
            #ax.set_yticks([x for x in range(3, 6)])
            # ax.set_xticks([x for x in range(1, 9)])
        plt.show()

    def test_incument_racing(self):
        """ Test racing bohb trajectory """
        def generator_to_dict(gen):
            return_dict = {'config_ids': [],
                           'times_finished': [],
                           'budgets': [],
                           'losses': []}
            last_run = None
            for r in gen:
                if last_run is None or last_run is not r:
                    return_dict['config_ids'].append(r.config_id)
                    return_dict['times_finished'].append(r.time_stamps['finished'])
                    return_dict['budgets'].append(r.budget)
                    return_dict['losses'].append(r.loss)
                last_run = r
            return return_dict

        all_runs_1 = [Run((0, 0, 1), 1, 3, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 1.0},
                          {}),
                      Run((0, 0, 2), 1, 6, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 2.0},
                          {}),
                      Run((0, 0, 2), 2, 3, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 3.0},
                          {}),
                      Run((0, 0, 1), 2, 4, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 4.0},
                          {}),
                      ]

        inc_dict = generator_to_dict(_compute_trajectory_racing(all_runs_1, budgets=[1, 2, 3]))
        result_run_1 = {'config_ids':
                            [(0,0,1), (0,0,2), ],
                        'times_finished':
                            [1.0,      4.0],
                        'budgets':
                            [1.0,      2.0],
                        'losses':
                            [3,        3]}

        self.assertEqual(result_run_1, inc_dict)
        if self.plot:
            self.plot_incs(inc_dict)

        all_runs_2 = [Run((0, 0, 1), 1, 3, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 1.0},
                          {}),
                      Run((0, 0, 2), 1, 6, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 2.0},
                          {}),
                      Run((0, 0, 2), 2, 5, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 3.0},
                          {}),
                      Run((0, 0, 2), 3, 4, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 4.0},
                          {}),
                      Run((0, 0, 1), 2, 6, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 5.0},
                          {}),
                      Run((0, 0, 1), 3, 6, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 6.0},
                          {}),
                      Run((0, 0, 3), 1, 2, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 7.0},
                          {}),
                      ]

        inc_dict = generator_to_dict(_compute_trajectory_racing(all_runs_2, budgets=[1, 2, 3]))
        result_run_2 = {'config_ids':
                            [(0,0,1), (0,0,2)],
                        'times_finished':
                            [1.0,      5.0],
                        'budgets':
                            [1.0,      3.0],
                        'losses':
                            [3,        4]}
        self.assertEqual(result_run_2, inc_dict)
        if self.plot:
            self.plot_incs(inc_dict)

        all_runs_3 = [Run((0, 0, 1), 1, 3, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 1.0},
                          {}),
                      Run((0, 0, 2), 1, 10, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 2.0},
                          {}),
                      Run((0, 0, 2), 2, 10, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 3.0},
                          {}),
                      Run((0, 0, 2), 3, 10, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 4.0},
                          {}),
                      Run((0, 0, 2), 4, 1, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 5.0},
                          {}),
                      Run((0, 0, 3), 1, 5, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 6.0},
                          {}),
                      Run((0, 0, 3), 2, 5, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 7.0},
                          {}),
                      Run((0, 0, 3), 3, 5, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 8.0},
                          {}),
                      Run((0, 0, 1), 2, 7, {},
                          {'submitted': 0.0, 'started': 0.0, 'finished': 9.0},
                          {}),
                      ]

        inc_dict = generator_to_dict(_compute_trajectory_racing(all_runs_3, budgets=[1, 2, 3, 4]))
        result_run_3 = {'config_ids':
                            [(0,0,1), (0,0,2)],
                        'times_finished':
                            [1.0,      9.0],
                        'budgets':
                            [1.0,      4.0],
                        'losses':
                            [3,        1]}
        self.assertEqual(result_run_3, inc_dict)
        if self.plot:
            self.plot_incs(inc_dict)

    def test_incumbent_trajectory(self):
        """ Load example result and check incumbent_trajectory generation for general errors (whitebox-test)"""
        result = logged_results_to_HBS_result(self.result_path)

        # All budgets
        traj = get_incumbent_trajectory(result, result.HB_config['budgets'], racing=True)
        traj = get_incumbent_trajectory(result, result.HB_config['budgets'], racing=False)

        # Single budgets
        traj = get_incumbent_trajectory(result, [result.HB_config['budgets'][0]], racing=True)
        traj = get_incumbent_trajectory(result, [result.HB_config['budgets'][0]], racing=False)
