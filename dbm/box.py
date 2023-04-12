import numpy as np

class Box():
    """
    A class for creating a 3D box object, which represents a simulation box in molecular dynamics. The Box class is used to:
    - store the size of the simulation box
    - deal with periodic boundary conditions
    - generate subboxes that improve computational efficiency

    Public Methods:
    - __init__(file, cutoff): Initializes the box object with the given parameters.
    - get_box_dim(file): Reads the box dimensions from the last line in the gro file and returns a 3x3 array.
    - get_subbox_size(cutoff): Returns the size of a subbox for the given cutoff value.
    - subbox(pos): Returns a tuple representing the subbox index for a given position.
    - subbox_range(num, max): Returns a list of indices representing the range of subboxes for a given subbox index and its maximum value.
    - nn_subboxes(sb): Returns a list of neighboring subboxes for a given subbox index.
    - empty_subbox_dict(): Returns a dictionary with all subboxes as keys and empty lists as values.
    - move_inside(pos): Returns a position vector inside the box.
    - diff_vec(diff_vec): Returns a difference vector between two positions inside the box.
    - diff_vec_batch(diff_vec): Returns a batch of difference vectors between positions inside the box.
    - get_vol(): Returns the volume of the box.

    Class Properties:
    - dim: A 3x3 array representing the dimensions of the box.
    - dim_inv: The inverse of the dim property.
    - v1, v2, v3: Column vectors representing the basis vectors of the box.
    - volume: The volume of the box.
    - center: The center point of the box.
    - max_v1_sub, max_v2_sub, max_v3_sub: The maximum number of subboxes in each direction.
    - subbox_size: The size of a subbox.
    - subbox_size_inv: The inverse of the subbox_size property.
    """

    def __init__(self, file, cutoff):
        """
        Constructor of the Box class

        Args:
        - file: The gro file containing the box dimensions.
        - cutoff: The cutoff distance for dividing the box into subboxes.
        """

        # Getting the box dimensions from the file
        self.dim = self.get_box_dim(file)

        # Calculating the inverse of the box dimensions matrix
        self.dim_inv = np.linalg.inv(self.dim)

        # Extracting the three box vectors v1, v2, v3 that span the simulation box
        self.v1 = self.dim[:, 0]
        self.v2 = self.dim[:, 1]
        self.v3 = self.dim[:, 2]

        # Calculating the maximum number of sub-boxes in each direction
        self.max_v1_sub = int(self.v1[0] / cutoff)
        self.max_v2_sub = int(self.v2[1] / cutoff)
        self.max_v3_sub = int(self.v3[2] / cutoff)

        # Calculating the volume of the simulation box
        self.volume = self.get_vol()

        # Calculating the center point of the box
        self.center = 0.5*self.v1 + 0.5*self.v2 + 0.5*self.v3

        # Calculating the size of each sub-box
        self.subbox_size = self.get_subbox_size()

        # Calculating the inverse of the sub-box size matrix
        self.subbox_size_inv = np.linalg.inv(self.subbox_size)

    def get_box_dim(self, file):
        """
        Reads the box dimensions from the last line in the gro file and returns a 3x3 array.

        Args:
        - file: A string representing the file path to the gro file containing the box dimensions.

        Returns:
        - dim: A 3x3 array representing the dimensions of the box.
        """

        f_read = open(file, "r")
        bd = np.array(f_read.readlines()[-1].split(), np.float32)
        f_read.close()
        bd = list(bd)
        for n in range(len(bd), 10):
            bd.append(0.0)
        dim = np.array([[bd[0], bd[5], bd[7]],
                                 [bd[3], bd[1], bd[8]],
                                 [bd[4], bd[6], bd[2]]])
        return dim

    def get_subbox_size(self):
        """
        Returns the size of a subbox for the given cutoff value.

        Args:
        - cutoff: A float representing the cutoff distance for dividing the box into subboxes.

        Returns:
        - subbox: A 3x3 array representing the size of each subbox.
        """

        v1_scaled = self.v1
        v1_scaled = v1_scaled / self.max_v1_sub
        v2_scaled = self.v2
        v2_scaled = v2_scaled / self.max_v2_sub
        v3_scaled = self.v3
        v3_scaled = v3_scaled / self.max_v3_sub
        subbox = np.array([[v1_scaled[0], v2_scaled[0], v3_scaled[0]],
                            [v1_scaled[1], v2_scaled[1], v3_scaled[1]],
                            [v1_scaled[2], v2_scaled[2], v3_scaled[2]]])
        return subbox

    def subbox(self, pos):
        """
        Given a position in space, return the sub-box index in which the position belongs.

        Parameters:
        pos (numpy array): the position in space (x,y,z) coordinates

        Returns:
        tuple: the sub-box index (a,b,c) to which the position belongs
        """

        f = np.dot(self.subbox_size_inv, pos)
        f = f.astype(int)
        return tuple(f)

    def subbox_range(self, num, max):
        """
        Given a number and a maximum value, return a list of the valid indices of neighboring sub-boxes.

        Parameters:
        num (int): the current sub-box index value
        max (int): the maximum sub-box index value

        Returns:
        list: a list of neighboring sub-box indices
        """

        if max == 0:
            range = [0]
        elif num == max:
            range = [num-1, num, 0]
        elif num == 0:
            range = [max, num, num+1]
        else:
            range = [num-1, num, num+1]
        return range

    def nn_subboxes(self, sb):
        """
        Given a sub-box index, return a list of the neighboring sub-box indices.

        Parameters:
        sb (tuple): the sub-box index (a,b,c) for which to find neighbors

        Returns:
        list: a list of neighboring sub-box indices
        """

        subboxes = []

        for a in self.subbox_range(sb[0], self.max_v1_sub-1):
            for b in self.subbox_range(sb[1], self.max_v2_sub-1):
                for c in self.subbox_range(sb[2], self.max_v3_sub-1):
                    subboxes.append((a,b,c))
        return list(set(subboxes))

    def empty_subbox_dict(self):
        """
        Return an empty dictionary with keys as all possible sub-box indices and values as empty lists.

        Returns:
        dict: a dictionary with keys as sub-box indices and values as empty lists
        """

        keys = []
        for a in range(0, self.max_v1_sub):
            for b in range(0, self.max_v2_sub):
                for c in range(0, self.max_v3_sub):
                    keys.append((a,b,c))
        subbox_dict = dict([(key, []) for key in keys])
        return subbox_dict


    def move_inside(self, pos):
        """
        Given a position in space, return the position within the periodic box.

        Parameters:
        pos (numpy array): the position in space (x,y,z) coordinates

        Returns:
        numpy array: the position in space within the periodic box
        """

        f = np.dot(self.dim_inv, pos)
        g = f - np.floor(f)
        new_pos = np.dot(self.dim, g)
        return new_pos

    def diff_vec(self, diff_vec):
        """
        Given a vector, return the vector's position within the periodic box.

        Parameters:
        diff_vec (numpy array): the vector (dx,dy,dz) to adjust for periodic boundary conditions

        Returns:
        numpy array: the adjusted vector (dx',dy',dz') with respect to the periodic box
        """

        diff_vec = diff_vec + self.center
        diff_vec = self.move_inside(diff_vec)
        diff_vec = diff_vec - self.center
        return diff_vec

    def diff_vec_batch(self, diff_vec):
        """
        Given a list of positions inside the box, return a batch of difference vectors between them.

        Parameters:
        diff_vec (numpy array): a list of positions in space (x,y,z) coordinates

        Returns:
        numpy array: a batch of difference vectors with respect to the periodic box
        """

        diff_vec = np.swapaxes(diff_vec, 0, 1)
        diff_vec = diff_vec + self.center[:, np.newaxis]
        diff_vec = self.move_inside(diff_vec)
        diff_vec = diff_vec - self.center[:, np.newaxis]
        diff_vec = np.swapaxes(diff_vec, 0, 1)
        return diff_vec

    def get_vol(self):
        """
        Calculate the volume of the simulation box.

        Returns:
        float: the volume of the box
        """
        norm1 = np.sqrt(np.sum(np.square(self.v1)))
        norm2 = np.sqrt(np.sum(np.square(self.v2)))
        norm3 = np.sqrt(np.sum(np.square(self.v3)))

        cos1 = np.sum(self.v2 * self.v3) / (norm2 * norm3)
        cos2 = np.sum(self.v1 * self.v3) / (norm1 * norm3)
        cos3 = np.sum(self.v1 * self.v2) / (norm1 * norm2)
        v = norm1*norm2*norm3 * np.sqrt(1-np.square(cos1)-np.square(cos2)-np.square(cos3)+2*np.sqrt(cos1*cos2*cos3))
        return v


