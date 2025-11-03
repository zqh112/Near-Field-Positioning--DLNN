import numpy as np

class ULA:
    def __init__(self, num_of_antennas: int, antenna_spacing: int,
                 centre_coordinate: np.ndarray):
        '''
        description:
        param {*} self
        param {int} num_of_antennas
        param {int} antenna_spacing
        param {np} centre_coordinate
        return {*}
        '''
        self.num_of_antennas = num_of_antennas
        self.centre_coordinate = centre_coordinate
        self.antenna_spacing = antenna_spacing
        self.aperture = antenna_spacing * (num_of_antennas - 1)
        self.element_coordinate = self._get_coordinate()

    def _get_coordinate(self):
        element_coordinate = np.vstack([np.linspace(-0.5 * (self.num_of_antennas - 1) * self.antenna_spacing,
                                                    0.5 * (self.num_of_antennas - 1) * self.antenna_spacing,
                                                    num=self.num_of_antennas, endpoint=True) + self.centre_coordinate[
                                            0],
                                        np.zeros(self.num_of_antennas) + self.centre_coordinate[1],
                                        np.zeros(self.num_of_antennas) + self.centre_coordinate[2]])
        return element_coordinate

if __name__ == '__main__':
    main()