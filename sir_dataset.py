import numpy as np
import matplotlib.pyplot as plt
import random
import progressbar
import imageio
import pickle
import scipy.ndimage


STATE_SUSCEPTIBLE = 1
STATE_INFECTED = 2
STATE_RECOVERED = 3
COLOR_SUSCEPTIBLE = (0.0, 0.0, 1.0)
COLOR_INFECTED = (1.0, 0.0, 0.0)
COLOR_RECOVERED = (0.0, 1.0, 0.0)


class SIRSimulator:
    """ Simulates a epidemic. """


    def __init__(self, rows, columns, sequence_length):
        """ Initializes the simulator. """

        self.rows = rows
        self.columns = columns
        self.sequence_length = sequence_length


    def generate_dataset(self, size, dataset_type, split_ratio, initially_infected, time_infection, time_recover, transmission_probability, average_contacts):
        """ Generates a full dataset. This includes train, validate and test. """

        print("Generating data-set...")

        # Make sure that all the parameters are list. For random.
        self.initially_infected_list = self.ensure_list(initially_infected)
        self.time_infection_list = self.ensure_list(time_infection)
        self.time_recover_list = self.ensure_list(time_recover)
        self.transmission_probability_list = self.ensure_list(transmission_probability)
        self.average_contacts_list = self.ensure_list(average_contacts)

        # Simulate with random choices.
        bar = progressbar.ProgressBar(max_value=size)
        input_data = []
        output_data = []
        for index in range(size):
            initially_infected = random.choice(self.initially_infected_list)
            time_infection = random.choice(self.time_infection_list)
            time_recover = random.choice(self.time_recover_list)
            transmission_probability = random.choice(self.transmission_probability_list)
            average_contacts = random.choice(self.average_contacts_list)

            states = self.simulate(initially_infected, time_infection, time_recover, transmission_probability, average_contacts)

            if dataset_type == "states":
                input_element = states
            elif dataset_type == "counts":
                input_element = states_to_counts(states)
            else:
                raise Exception("Unknown type:", dataset_type)
            input_data.append(input_element)

            output_element = np.array((initially_infected, time_infection, time_recover, transmission_probability, average_contacts))
            output_data.append(output_element)
            bar.update(index)
        bar.finish()

        # Split into train, validate and test.
        print("Splitting...")
        split_ratio = np.array(split_ratio.split(":")).astype("float32")
        split_ratio /= np.sum(split_ratio)
        first_index = int(split_ratio[0] * size)
        second_index = int((split_ratio[0] + split_ratio[1] ) * size)
        train_input = np.array(input_data[0:first_index])
        train_output = np.array(output_data[0:first_index])
        validate_input = np.array(input_data[first_index:second_index])
        validate_output = np.array(output_data[first_index:second_index])
        test_input = np.array(input_data[second_index:])
        test_output = np.array(output_data[second_index:])

        # Done so far.
        return (train_input, train_output), (validate_input, validate_output), (test_input, test_output)


    def ensure_list(self, element):
        """ Ensures that a provided element is a list. """

        if type(element) is list:
            return element
        else:
            return [element]


    def simulate(self, initially_infected, time_infection, time_recover, transmission_probability, average_contacts):
        """ Runs a full simulation with the provided set of parameters. """

        grid = self.create_grid(initially_infected, time_infection)

        states = []
        state = self.grid_to_state(grid)
        states.append(state)
        for step in range(self.sequence_length - 1):
            grid = self.do_simulation_step(grid, time_infection, time_recover, transmission_probability, average_contacts)
            state = self.grid_to_state(grid)
            states.append(state)

        states = np.array(states)
        return states


    def create_grid(self, initially_infected, time_infection):
        """ Creates a grid. """

        grid = np.zeros((self.rows, self.columns, 2))

        # Create the initial grid.
        for row in range(self.rows):
            for column in range(self.columns):
                status = STATE_SUSCEPTIBLE
                time = 0.0
                grid[row, column] = (status, time)

        # Infect some cells.
        infected = 0
        while infected < initially_infected:
            random_column = np.random.randint(self.columns)
            random_row = np.random.randint(self.rows)
            if grid[random_row, random_column, 0] == STATE_SUSCEPTIBLE:
                grid[random_row, random_column, 0] = STATE_INFECTED
                grid[random_row, random_column, 1] = time_infection
                infected += 1

        return grid


    def grid_to_state(self, grid):
        """ Converts a grid to a state. """

        state = np.array(grid[:,:,0]) # Drip the last element, which is times.
        return state


    def do_simulation_step(self, grid, time_infection, time_recover, transmission_probability, average_contacts):
        """ Does one step of the simulation. """

        for row in range(grid.shape[0]):
            for column in range(grid.shape[1]):

                # Let people recover.
                if grid[row, column, 0] == STATE_RECOVERED:
                    if grid[row, column, 1] > 0.0:
                        grid[row, column, 1] -= 1.0
                    else:
                        grid[row, column, 0] = STATE_SUSCEPTIBLE
                        grid[row, column, 1] = 0.0

                # Heal.
                if grid[row, column, 0] == STATE_INFECTED:
                    if grid[row, column, 1] > 0.0:
                        grid[row, column, 1] -= 1.0
                    else:
                        grid[row, column, 0] = STATE_RECOVERED
                        grid[row, column, 1] = time_recover

                # Randomly infect.
                else:
                    for _ in range(average_contacts):
                        random_row = np.random.randint(grid.shape[0])
                        random_column = np.random.randint(grid.shape[1])
                        if grid[row, column, 0] == grid[random_row, random_column, 0]:
                            pass
                        elif grid[row, column, 0] == STATE_INFECTED and grid[random_row, random_column, 0] == STATE_SUSCEPTIBLE:
                            if random.uniform(0.0, 1.0) < transmission_probability:
                                grid[random_row, random_column, 0] = STATE_INFECTED
                                grid[random_row, random_column, 1] = time_infection
                        elif grid[row, column, 0] == STATE_SUSCEPTIBLE and grid[random_row, random_column, 0] == STATE_INFECTED:
                            if random.uniform(0.0, 1.0) < transmission_probability:
                                grid[row, column, 0] = STATE_INFECTED
                                grid[row, column, 1] = time_infection

        return grid


def render_states(states, path):
    """ Renders a sequence of states into an animated gif. """

    print("Rendering states to ", path, "...")
    images = [(state_to_image(state) * 255).astype("uint8") for state in states]
    images = [scipy.ndimage.zoom(image, (4, 4, 1), order=0) for image in images]
    imageio.mimsave(path, images, duration = 0.1)


def state_to_image(state):
    """ Converts a state to a RGB-image. """

    image = np.zeros((state.shape[0], state.shape[1], 3))
    for row in range(state.shape[0]):
        for column in range(state.shape[1]):
            if state[row, column] == STATE_SUSCEPTIBLE:
                color = COLOR_SUSCEPTIBLE
            elif state[row, column] == STATE_INFECTED:
                color = COLOR_INFECTED
            elif state[row, column] == STATE_RECOVERED:
                color = COLOR_RECOVERED
            image[row, column] = color

    return image


def states_to_counts(states):
    """ Converts a sequence of states to a sequence of counts. """

    result = []
    for state in states:
        unique, counts = np.unique(state, return_counts=True)
        dictionary = dict(zip(unique, counts))
        susceptible_count = dictionary.get(STATE_SUSCEPTIBLE, 0)
        infected_count = dictionary.get(STATE_INFECTED, 0)
        recovered_count = dictionary.get(STATE_RECOVERED, 0)
        result.append((susceptible_count, infected_count, recovered_count))

    result = np.array(result)
    return result


def render_dataset(dataset):
    """ Renders a full dataset. """

    (train_input, train_output), (validate_input, validate_output), (test_input, test_output) = dataset

    render_subset(train_input, "train")
    render_subset(validate_input, "validate")
    render_subset(train_input, "test")


def render_subset(data_input, name):
    """ Renders a subset of a dataset. """

    for index, input_element in enumerate(data_input):
        path = "diagram-{}-{}.png".format(name, index)
        render_counts(path, input_element)


def render_counts(path, data):
    """ Renders a sequence of counts. """

    print("Rendering counts to", path, "...")

    susceptible_counts = np.array(data[:, 0])
    infected_counts = np.array(data[:, 1])
    recovered_counts = np.array(data[:, 2])

    population = susceptible_counts[0] + infected_counts[0] + recovered_counts[0]
    susceptible_counts = susceptible_counts / population
    infected_counts = infected_counts / population
    recovered_counts = recovered_counts / population

    time_steps = range(len(susceptible_counts))

    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(time_steps, susceptible_counts, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(time_steps, infected_counts, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(time_steps, recovered_counts, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    fig.savefig(path)
    plt.close(fig)
    plt.clf()


def save_dataset(dataset, path):
    """ Saves a dataset. """

    print("Saving dataset to ", path, "..." )
    pickle.dump(dataset, open(path, "wb"))


def load_dataset(path):
    """ Loads a dataset. """

    print("Loading dataset from ", path, "..." )
    dataset = pickle.load(open(path, "rb"))
    (train_input, train_output), (validate_input, validate_output), (test_input, test_output) = dataset
    return dataset


def print_dataset_statistics(dataset):
    """ Prints the statistics of a dataset. """

    (train_input, train_output), (validate_input, validate_output), (test_input, test_output) = dataset

    print("train_input", train_input.shape)
    print("train_output", train_output.shape)
    print("validate_input", validate_input.shape)
    print("validate_output", validate_output.shape)
    print("test_input", test_input.shape)
    print("test_output", test_output.shape)



def test_method_one():

    sir_simulator = SIRSimulator(20, 20, 50)

    size = 10
    dataset_type = "counts"

    dataset = sir_simulator.generate_dataset(
        size = size,
        split_ratio = "7:1:2",
        dataset_type = dataset_type,
        initially_infected = [5, 10, 20, 40],
        time_infection = [1, 2, 4, 8],
        time_recover = [2, 4, 8, 16],
        transmission_probability = [0.1, 0.2, 0.4, 0.8],
        average_contacts = [2, 4, 8, 16]
        )

    print_dataset_statistics(dataset)

    dataset_path = "dataset-{}-{}.p".format(dataset_type, size)
    save_dataset(dataset, dataset_path)


def test_method_two():

    size = 10
    dataset_type = "counts"

    dataset_path = "dataset-{}-{}.p".format(dataset_type, size)
    dataset = load_dataset(dataset_path)

    render_dataset(dataset)


def test_method_three():
    sir_simulator = SIRSimulator(50, 50, 100)

    initially_infected = 5
    time_infection = 2
    time_recover = 4
    transmission_probability = 0.4
    average_contacts = 2
    states = sir_simulator.simulate(initially_infected, time_infection, time_recover, transmission_probability, average_contacts)

    render_states(states, "output-1.gif")
    counts = states_to_counts(states)
    render_counts("output-1.png", counts)

    initially_infected = 5
    time_infection = 16
    time_recover = 4
    transmission_probability = 0.3
    average_contacts = 8
    states = sir_simulator.simulate(initially_infected, time_infection, time_recover, transmission_probability, average_contacts)

    render_states(states, "output-2.gif")
    counts = states_to_counts(states)
    render_counts("output-2.png", counts)


def test_method_four():
    dataset_path = "dataset-counts-10.p"
    (train_input, train_output), _, _ = load_dataset(dataset_path)
    print("train_input:", train_input[0][:10])
    print("train_output:", train_output[0])


if __name__ == "__main__":
    #test_method_one()
    #test_method_two()
    #test_method_three()
    test_method_four()
