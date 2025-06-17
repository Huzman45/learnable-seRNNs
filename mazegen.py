# Based on the Capsule for Achterberg & Akarca, et al:
# Spatially-embedded recurrent neural networks reveal
# widespread structural and functional neuroscience findings
# https://codeocean.com/capsule/2879348/tree/v2


import numpy as np
import tensorflow as tf
import pandas as pd

np.random.seed(18229)
tf.random.set_seed(94892)


class mazeGeneratorI:
    """
    Objects of the mazeGeneratorI class can create numpy and tf datasets of the first choice of the maze task.
    Task structure:
        Goal presentation, followed by delay period, followed by choice options.
    Response:
        One response required from agent at end of episode. Direction (Left, Up, Right, Down) of first step.
    Encoding:
        Both observations and labels are OneHot encoded.
    Usage:
        The two only function a user should need to access are "construct_numpy_data" and "construct_tf_data"
    Options:
        Both data construction methods have an option to shuffle the labels of data.
        The numpy data construction method allows to also return the maze identifiers.
    """

    def __init__(
        self, goal_presentation_steps, delay_steps, choices_presentation_steps
    ):
        self.version = "v1.2.0"

        # Import variables defining episode
        self.goal_presentation_steps = goal_presentation_steps
        self.delay_steps = delay_steps
        self.choices_presentation_steps = choices_presentation_steps

        # Construct mazes dataframe
        # Add encoded versions of the goal / choices presentations and the next step response
        self.mazesdf = self.import_maze_dic()
        self.mazesdf["Goal_Presentation"] = self.mazesdf["goal"].map(
            {
                7: np.concatenate((np.array([1, 0, 0, 0]), np.repeat(0, 4))),
                9: np.concatenate((np.array([0, 1, 0, 0]), np.repeat(0, 4))),
                17: np.concatenate((np.array([0, 0, 1, 0]), np.repeat(0, 4))),
                19: np.concatenate((np.array([0, 0, 0, 1]), np.repeat(0, 4))),
            }
        )
        self.mazesdf["Choices_Presentation"] = self.mazesdf["ChoicesCategory"].map(
            lambda x: self.encode_choices(x=x)
        )
        self.mazesdf["Step_Encoded"] = self.mazesdf["NextFPmap"].map(
            lambda x: self.encode_next_step(x=x)
        )

    def construct_numpy_data(
        self, number_of_problems, return_maze_identifiers=False, np_shuffle_data=False
    ):
        # Create a new column which hold the vector for each problem
        self.mazesdf["Problem_Vec"] = self.mazesdf.apply(
            lambda x: self.create_problem_observation(
                row=x,
                goal_presentation_steps=self.goal_presentation_steps,
                delay_steps=self.delay_steps,
                choices_presentation_steps=self.choices_presentation_steps,
            ),
            axis=1,
        )
        # Set a random order of maze problems for the current session
        self.mazes_order = np.random.randint(0, 8, number_of_problems)

        # Create vectors, holding observations and labels
        session_observation = np.array([])
        session_labels = np.array([])
        for i in self.mazes_order:
            session_observation = np.append(
                session_observation, self.mazesdf.iloc[i]["Problem_Vec"]
            )
            session_labels = np.append(
                session_labels, self.mazesdf.iloc[i]["Step_Encoded"]
            )

        # Reshape vectors to fit network observation and response space
        session_length = (
            self.goal_presentation_steps
            + self.delay_steps
            + self.choices_presentation_steps
        )
        session_observation = np.reshape(
            session_observation, (-1, session_length, 8)
        ).astype("float32")
        session_labels = np.reshape(session_labels, (-1, 4)).astype("float32")

        # If np_shuffle_data == 'Labels, the order of labels is shuffled to randomise correct answers
        if np_shuffle_data == "Labels":
            shuffle_generator = np.random.default_rng(38446)
            shuffle_generator.shuffle(session_labels, axis=0)

        # If return_maze_identifiers == 'IDs', return the array with maze IDs alongside the regular returns (observations, labels)
        if return_maze_identifiers == "IDs":
            return session_observation, session_labels, self.mazes_order

        return session_observation, session_labels

    def construct_tf_data(self, number_of_problems, batch_size, tf_shuffle_data=False):
        # Create dataset as described by numpy dataset function and transform it into a TF dataset
        npds, np_labels = self.construct_numpy_data(
            number_of_problems=number_of_problems, np_shuffle_data=tf_shuffle_data
        )
        tfdf = tf.data.Dataset.from_tensor_slices((npds, np_labels))
        tfdf = tfdf.batch(batch_size)
        return tfdf

    def reset_construction_params(
        self, goal_presentation_steps, delay_steps, choices_presentation_steps
    ):
        self.goal_presentation_steps = goal_presentation_steps
        self.delay_steps = delay_steps
        self.choices_presentation_steps = choices_presentation_steps

    def import_maze_dic(self, mazeDic=None):
        if mazeDic == None:
            # Set up dataframe with first choices of maze task
            # The dictionary was generated using MazeMetadata.py (v1.0.0) and the following call:
            # mazes.loc[(mazes['Nsteps']==2)&(mazes['ChoiceNo']=='ChoiceI')][['goal','ChoicesCategory','NextFPmap']].reset_index(drop=True).to_dict()
            self.mazesDic = {
                "goal": {0: 9, 1: 9, 2: 19, 3: 17, 4: 17, 5: 7, 6: 19, 7: 7},
                "ChoicesCategory": {
                    0: "ul",
                    1: "rd",
                    2: "ld",
                    3: "rd",
                    4: "ul",
                    5: "ur",
                    6: "lr",
                    7: "lr",
                },
                "NextFPmap": {
                    0: "u",
                    1: "r",
                    2: "d",
                    3: "d",
                    4: "l",
                    5: "u",
                    6: "r",
                    7: "l",
                },
            }
        else:
            self.mazesDic = mazeDic

        # Create and return dataframe
        return pd.DataFrame(self.mazesDic)

    def encode_choices(self, x):
        # Helper function to create the observation vector for choice periods
        choices_sec = np.repeat(0, 4)
        choicesEncoding = pd.Series(list(x))
        choicesEncoding = choicesEncoding.map({"l": 1, "u": 2, "r": 3, "d": 4})
        for encodedChoice in choicesEncoding:
            choices_sec[encodedChoice - 1] = 1
        return np.concatenate((np.repeat(0, 4), choices_sec))

    def encode_next_step(self, x):
        # Helper function to change the response / action to a OneHot encoded vector
        step_sec = np.repeat(0, 4)
        stepEncoding = pd.Series(list(x))
        stepEncoding = stepEncoding.map({"l": 1, "u": 2, "r": 3, "d": 4})
        for encodedStep in stepEncoding:
            step_sec[encodedStep - 1] = 1
        return step_sec

    def create_problem_observation(
        self, row, goal_presentation_steps, delay_steps, choices_presentation_steps
    ):
        # Helper function to create one vector describing the entire outline of one maze problem (Goal presentation, Delay Period, and Choices Presentation)
        goal_vec = np.tile(row["Goal_Presentation"], goal_presentation_steps)
        delay_vec = np.tile(np.repeat(0, 8), delay_steps)
        choices_vec = np.tile(row["Choices_Presentation"], choices_presentation_steps)
        problem_vec = np.concatenate((goal_vec, delay_vec, choices_vec))
        return problem_vec

    def __repr__(self):
        return "\n".join(
            [
                f"Maze DataSet Generator",
                f"Goal Presentation Steps: {self.goal_presentation_steps}",
                f"Delay Steps: {self.delay_steps}",
                f"Choices Presentation Steps: {self.choices_presentation_steps}",
            ]
        )
