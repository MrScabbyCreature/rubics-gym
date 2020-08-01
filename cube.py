import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d


face_to_num_map = {
    'L': 0, # left
    'R': 1, # right
    'F': 2, # front
    'B': 3, # back
    'U': 4, # up
    'D': 5, # down
}

face_to_color_map = {
    # this is to map to an actual rubicks cube
    'L': 'O', 
    'R': 'R', 
    'F': 'W', 
    'B': 'Y', 
    'U': 'B', 
    'D': 'G',
}

num_to_face_map = {face_to_num_map[key]: key for key in face_to_num_map}


CLOCK=1
ANTICLOCK=0

class Face:
    def __init__(self, n,  init_value=0):
        self.face = np.zeros((n, n), np.uint8)
        self.face[...] = init_value
        self.n = n

    def get_face_map_array(self):
        return np.vectorize(num_to_face_map.get)(self.face)

    def get_color_map_array(self):
        return np.vectorize(face_to_color_map.get)(np.vectorize(num_to_face_map.get)(self.face))
        
    def __repr__(self):
        temp = self.get_letter_map_array()
        representation = "\n".join([" ".join(list(map(str, temp[i]))) for i in range(self.n)])
        return representation
    
    def __getitem__(self, index):
        return self.face[index]

    def __setitem__(self, index, value):
        self.face[index] = value

class Cube:
    def __init__(self, n=3):
        self.faces = {}
        for letter in face_to_num_map:
            self.faces[letter] = Face(n, face_to_num_map[letter])
        self.n = n
        
    def print_cube(self,):
        ''' this is the pretty print standard form of the cube '''
        # print U
        for i in range(self.n):
            print(" ".join(
                [" "] * self.n + [" "] +
                list(map(str, self.faces['U'].get_face_map_array()[i]))
            ))
        print()

        # print L, F, R, B
        for i in range(self.n):
            print(" ".join(
                list(map(str, self.faces['L'].get_face_map_array()[i])) + [" "] +
                list(map(str, self.faces['F'].get_face_map_array()[i])) + [" "] +
                list(map(str, self.faces['R'].get_face_map_array()[i])) + [" "] +
                list(map(str, self.faces['B'].get_face_map_array()[i]))
            ))
        print()

        # print U
        for i in range(self.n):
            print(" ".join(
                [" "] * self.n + [" "] +
                list(map(str, self.faces['D'].get_face_map_array()[i]))
            ))

    def print_cube_with_colors(self):
        ''' this is the pretty print standard form of the cube '''
        # print U
        for i in range(self.n):
            print(" ".join(
                [" "] * self.n + [" "] +
                list(map(str, self.faces['U'].get_color_map_array()[i]))
            ))
        print()

        # print L, F, R, B
        for i in range(self.n):
            print(" ".join(
                list(map(str, self.faces['L'].get_color_map_array()[i])) + [" "] +
                list(map(str, self.faces['F'].get_color_map_array()[i])) + [" "] +
                list(map(str, self.faces['R'].get_color_map_array()[i])) + [" "] +
                list(map(str, self.faces['B'].get_color_map_array()[i]))
            ))
        print()

        # print U
        for i in range(self.n):
            print(" ".join(
                [" "] * self.n + [" "] +
                list(map(str, self.faces['D'].get_color_map_array()[i]))
            ))

    def rotate(self, face, direction=CLOCK, slice_dist=0):
        '''
        args
        ----
        face: int or str 
            The face to rotate. Should be a valid value.
        direction: int
            clockwise or anti-clockwise. Default is clockwise
        slice_dist: int
            for cubes greater than 3, slice_dist decides the slice to move. 
            The slice selected is 'slice_dist' distance from the face provided.
            The value for slice_dist <= (self.n - 1) / 2

        returns
        -------
        None
        '''
        if isinstance(face, int):
            assert face in face_to_num_map, f"{face} not a valid face"
            face = num_to_face_map[face]
        assert face in face_to_num_map, f"{face} not a valid face"
        assert direction in [CLOCK, ANTICLOCK], f"{direction} invalid. Clockwise: {CLOCK}, Anticlockwise: {ANTICLOCK}"
        assert slice_dist <= (self.n - 1) // 2, f"Max slice distance from face for cube of length {self.n} is {(self.n - 1)//2} but was passed {slice_dist}."

        # rotation logic
        if direction == CLOCK:
            self.faces[face][...] = np.rot90(self.faces[face][...], k=3)
        else:
            self.faces[face][...] = np.rot90(self.faces[face][...], k=1)
            
        F = self.faces['F'][...]
        B = self.faces['B'][...]
        U = self.faces['U'][...]
        D = self.faces['D'][...]
        L = self.faces['L'][...]
        R = self.faces['R'][...]

        if face == "F":
            if direction == CLOCK:
                temp = L[:, self.n-1-slice_dist].copy()
                L[:, self.n-1-slice_dist] = D[slice_dist, :].copy()
                D[slice_dist, :] = R[:, slice_dist][::-1].copy()
                R[:, slice_dist] = U[self.n-1-slice_dist, :].copy()
                U[self.n-1-slice_dist, :] = temp[::-1].copy()
            else:
                temp = L[:, self.n-1-slice_dist].copy()
                L[:, self.n-1-slice_dist] = U[self.n-1-slice_dist, :][::-1].copy()
                U[self.n-1-slice_dist, :] = R[:, slice_dist].copy() 
                R[:, slice_dist] = D[slice_dist, :][::-1].copy()
                D[slice_dist, :] = temp.copy()

        elif face == "L":
            if direction == CLOCK:
                temp = D[:, slice_dist].copy()
                D[:, slice_dist] = F[:, slice_dist].copy()
                F[:, slice_dist] = U[:, slice_dist].copy()
                U[:, slice_dist] = B[:, self.n - 1][::-1].copy()
                B[:, self.n - 1] = temp[::-1].copy()
            else:
                temp = D[:, slice_dist].copy()
                D[:, slice_dist] = B[:, self.n - 1][::-1].copy()
                B[:, self.n - 1] = U[:, slice_dist][::-1].copy()
                U[:, slice_dist] = F[:, slice_dist].copy()
                F[:, slice_dist] = temp.copy()

        elif face == "R":
            if direction == CLOCK:
                temp = U[:, self.n-1-slice_dist].copy()
                U[:, self.n-1-slice_dist] = F[:, self.n-1-slice_dist].copy()
                F[:, self.n-1-slice_dist] = D[:, self.n-1-slice_dist].copy()
                D[:, self.n-1-slice_dist] = B[:, slice_dist][::-1].copy()
                B[:, slice_dist] = temp[::-1].copy()
            else:
                temp = U[:, self.n-1-slice_dist].copy()
                U[:, self.n-1-slice_dist] = B[:, slice_dist][::-1].copy()
                B[:, slice_dist] = D[:, self.n-1-slice_dist][::-1].copy()
                D[:, self.n-1-slice_dist] = F[:, self.n-1-slice_dist].copy()
                F[:, self.n-1-slice_dist] = temp.copy()

        elif face == "U":
            if direction == CLOCK:
                temp = F[slice_dist, :].copy()
                F[slice_dist, :] = R[slice_dist, :].copy()
                R[slice_dist, :] = B[slice_dist, :].copy()
                B[slice_dist, :] = L[slice_dist, :].copy()
                L[slice_dist, :] = temp.copy()
            else:
                temp = F[slice_dist, :].copy()
                F[slice_dist, :] = L[slice_dist, :].copy()
                L[slice_dist, :] = B[slice_dist, :].copy()
                B[slice_dist, :] = R[slice_dist, :].copy()
                R[slice_dist, :] = temp.copy()

        elif face == "D":
            if direction == CLOCK:
                temp = F[self.n-1-slice_dist, :].copy()
                F[self.n-1-slice_dist, :] = L[self.n-1-slice_dist, :].copy()
                L[self.n-1-slice_dist, :] = B[self.n-1-slice_dist, :].copy()
                B[self.n-1-slice_dist, :] = R[self.n-1-slice_dist, :].copy()
                R[self.n-1-slice_dist, :] = temp.copy()
            else:
                temp = F[self.n-1-slice_dist, :].copy()
                F[self.n-1-slice_dist, :] = R[self.n-1-slice_dist, :].copy()
                R[self.n-1-slice_dist, :] = B[self.n-1-slice_dist, :].copy()
                B[self.n-1-slice_dist, :] = L[self.n-1-slice_dist, :].copy()
                L[self.n-1-slice_dist, :] = temp.copy()

        elif face == "B":
            if direction == CLOCK:
                temp = U[slice_dist, :].copy()
                U[slice_dist, :] = R[:, self.n-1-slice_dist].copy()
                R[:, self.n-1-slice_dist] = D[self.n-1-slice_dist, :][::-1].copy()
                D[self.n-1-slice_dist, :] = L[:, slice_dist].copy()
                L[:, slice_dist] = temp[::-1].copy()
            else:
                temp = U[slice_dist, :].copy()
                U[slice_dist, :] = L[:, slice_dist][::-1].copy()
                L[:, slice_dist] = D[self.n-1-slice_dist, :].copy()
                D[self.n-1-slice_dist, :] = R[:, self.n-1-slice_dist][::-1].copy()
                R[:, self.n-1-slice_dist] = temp.copy()
                
        self.faces['F'][...] = F
        self.faces['B'][...] = B
        self.faces['U'][...] = U
        self.faces['D'][...] = D
        self.faces['L'][...] = L
        self.faces['R'][...] = R

    def plot_cube(self):
        if not hasattr(self, "fig") or not plt.fignum_exists(100):
            self.fig = plt.figure(num=100)
            self.ax = self.fig.add_subplot(111, projection='3d')

        plot_color_map = {
            'R': 'red',
            'Y': 'yellow',
            'G': 'lawngreen',
            'B': 'mediumblue',
            'W': 'white',
            'O': 'orange'
        }

        F = self.faces['F'].get_color_map_array()
        B = self.faces['B'].get_color_map_array()
        R = self.faces['R'].get_color_map_array()
        L = self.faces['L'].get_color_map_array()
        U = self.faces['U'].get_color_map_array()
        D = self.faces['D'].get_color_map_array()

        # 'F'
        for i in range(self.n):
            for j in range(self.n):
                p = Rectangle((i, j), 1, 1)
                # p.set_color(plot_color_map[F[self.n - 1 - j, self.n - 1 - i]])
                p.set_color(plot_color_map[F[self.n - 1 - j, i]])
                p.set_edgecolor("black")
                self.ax.add_patch(p)
                art3d.pathpatch_2d_to_3d(p, z=0, zdir="y")

        # 'B'
        for i in range(self.n):
            for j in range(self.n):
                p = Rectangle((i, j), 1, 1)
                # p.set_color(plot_color_map[B[self.n - 1 - j, i]])
                p.set_color(plot_color_map[B[self.n - 1 - j, self.n - 1 - i]])
                p.set_edgecolor("black")
                self.ax.add_patch(p)
                art3d.pathpatch_2d_to_3d(p, z=self.n, zdir="y")

        # # 'R'
        for i in range(self.n):
            for j in range(self.n):
                p = Rectangle((i, j), 1, 1)
                p.set_color(plot_color_map[R[self.n - 1 - j, i]])
                p.set_edgecolor("black")
                self.ax.add_patch(p)
                art3d.pathpatch_2d_to_3d(p, z=self.n, zdir="x")

        # # 'L'
        for i in range(self.n):
            for j in range(self.n):
                p = Rectangle((i, j), 1, 1)
                p.set_color(plot_color_map[L[self.n - 1 - j, self.n - 1 -i]])
                p.set_edgecolor("black")
                self.ax.add_patch(p)
                art3d.pathpatch_2d_to_3d(p, z=0, zdir="x")

        # 'U'
        for i in range(self.n):
            for j in range(self.n):
                p = Rectangle((i, j), 1, 1)
                p.set_color(plot_color_map[U[self.n - 1 - j, i]])
                p.set_edgecolor("black")
                self.ax.add_patch(p)
                art3d.pathpatch_2d_to_3d(p, z=self.n, zdir="z")

        # 'D'
        for i in range(self.n):
            for j in range(self.n):
                p = Rectangle((i, j), 1, 1)
                p.set_color(plot_color_map[D[j, i]])
                p.set_edgecolor("black")
                self.ax.add_patch(p)
                art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")


        self.ax.set_xlim(0, self.n)
        self.ax.set_ylim(0, self.n)
        self.ax.set_zlim(0, self.n)
        plt.axis('off')
        plt.pause(0.01)
        # plt.show()

if __name__ == "__main__":
    import sys
    import time
    print(sys.argv)
    try:
        n = int(sys.argv[1])
    except:
        n = 3
    C = Cube(n)
    C.print_cube()
    letters = list(face_to_num_map.keys())
    np.random.shuffle(letters)
    C.print_cube_with_colors()
    C.plot_cube()
    
    print("\n")
    for letter in letters:
        print(letter)
        C.rotate(letter, direction=CLOCK, slice_dist=0)
        C.print_cube_with_colors()
        C.plot_cube()
        print("\n")

        time.sleep(0.2)

    print("-------------------------------------------------")
    plt.close()

    for letter in letters:
        print(letter)
        C.rotate(letter, direction=CLOCK, slice_dist=0)
        C.print_cube_with_colors()
        C.plot_cube()
        print("\n")

        time.sleep(0.2)
