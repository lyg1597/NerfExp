import numpy as np 
import torch 
import os 
import pickle 
import matplotlib.pyplot as plt 
import copy 

def plot_ellipsoid_2d(R, x0, num_points=1000, ax=None):
    """
    Plots a 2D ellipsoid defined by ||R(x - x0)|| <= 1.

    Parameters:
    - R: A 2x2 positive definite matrix.
    - x0: Center of the ellipsoid (array-like of length 2).
    - num_points: Number of points to use for plotting the ellipse.
    - ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
    """
    # Ensure R is a numpy array
    R = np.asarray(R)
    x0 = np.asarray(x0)

    # Ensure R is positive definite
    if not np.all(np.linalg.eigvals(R) > 0):
        raise ValueError("Matrix R must be positive definite.")

    # Compute the inverse of R
    R_inv = np.linalg.inv(R)

    # Parameter for the angle
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Unit circle points
    circle = np.array([np.cos(theta), np.sin(theta)])  # Shape: (2, num_points)

    # Map the unit circle to the ellipse
    ellipse = R_inv @ circle + x0.reshape(2, 1)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(ellipse[0, :], ellipse[1, :], 'b-', label='Ellipsoid Boundary')
    ax.plot(x0[0], x0[1], 'ro', label='Center')

    # Setting the axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Ellipsoid Visualization')
    # ax.legend()
    ax.axis('equal')
    ax.grid(True)

    if ax is None:
        plt.show()


def create_positive_definite_matrix(L_elements):
    # Initialize the neural network outputs to parameterize the lower-triangular matrix
    # L_elements = neural_network_output  # Should have n(n+1)/2 elements

    # Construct the lower-triangular matrix
    L = torch.zeros((5, 5))
    indices = torch.tril_indices(5, 5)
    L[indices[0], indices[1]] = L_elements

    # Ensure positive diagonal entries
    diag_indices = torch.arange(5)
    L[diag_indices, diag_indices] = torch.nn.functional.softplus(L[diag_indices, diag_indices])+0.1

    return L

def min_max_dimension_values(C, x0, i):
    """
    Computes the maximum and minimum values along coordinate axis x_i
    for the ellipsoid defined by ||C(x - x0)|| <= 1.

    Parameters:
    - C: (n x n) numpy array representing matrix C.
    - x0: (n,) numpy array representing the center x_0 of the ellipsoid.
    - i: Integer index (0-based) of the coordinate axis x_i.

    Returns:
    - x_i_max: Maximum value along coordinate x_i.
    - x_i_min: Minimum value along coordinate x_i.
    """
    # Compute A = Cᵗ * C
    A = C.T @ C

    # Number of dimensions
    n = C.shape[0]

    # Standard basis vector e_i (with a 1 at index i)
    e_i = np.zeros(n)
    e_i[i] = 1.0

    # Solve A z = e_i for z using Cholesky decomposition
    # Since A is symmetric positive definite, Cholesky decomposition is applicable
    try:
        L = np.linalg.cholesky(A)
    except:
        return x0[i], x0[i]
    # Solve L y = e_i
    y = np.linalg.solve(L, e_i)
    # Solve Lᵗ z = y
    z = np.linalg.solve(L.T, y)

    # Extract z_i
    z_i = z[i]

    # Ensure z_i is positive to avoid taking sqrt of negative number
    if z_i <= 0:
        raise ValueError(f"Computed z_i is non-positive: z_i = {z_i}")

    # Compute sqrt(z_i)
    sqrt_z_i = np.sqrt(z_i)

    # Compute maximum and minimum values along x_i
    x_i_max = x0[i] + sqrt_z_i
    x_i_min = x0[i] - sqrt_z_i

    return x_i_max, x_i_min


class PC(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PC, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.model(x)
    
def pre_process_data2(data):

    state_list = []
    trace_list = []
    E_list = []
    for i in range(len(data)):
        X0 = data[i][0]
        x, Ec, Er = X0
        state_list.append(x)
        traces = data[i][1]
        traces = np.reshape(traces,(-1,6))
        trace_list.append(traces)
        init = data[i][2]
        # for tmp in init:
        #     E_list.append(tmp[1])
        E_list.append(init[1])
    # Getting Model for center center model
    state_array = np.array(state_list)
    trace_array = np.array(trace_list).squeeze()
    E_array = np.array(E_list)

    return state_array, trace_array, E_array 

class PositionalEncoder(torch.nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(self, d_input: int, n_freqs: int, log_space: bool = False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.0 ** torch.linspace(0.0, self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(
                2.0**0.0, 2.0 ** (self.n_freqs - 1), self.n_freqs
            )

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, './data_pc/data_09-02-17-23.pickle')
    

    input_size = 5 
    hidden_size = 64
    output_size = 15

    alpha = 5
    gamma = 0.0001


    radius_encoder = PositionalEncoder(
        input_size, 4, True
    )
    model_radius = PC(radius_encoder.d_output, hidden_size, output_size)

    model_center = PC(5, 64, 5)

    optimizer_center = torch.optim.Adam(model_center.parameters(), lr=1e-4)
    optimizer_radius = torch.optim.Adam(model_radius.parameters(), lr=1e-4)

    with open(data_file_path,'rb') as f:
        data = pickle.load(f)

    state_array, trace_array, E_array = pre_process_data2(data)

    train_input = torch.FloatTensor(state_array[:,:5])
    train_label = torch.FloatTensor(trace_array[:,:5])

    epoch = 5000
    for i in range(epoch):
        data = train_input 
        label = train_label 
        res = model_center(data)
        loss_each = torch.linalg.norm(res-label, dim = 1)
        loss_center = loss_each.mean()
        optimizer_center.zero_grad()
        loss_center.backward()
        optimizer_center.step()  
        print(i, loss_center.item())

    model_center.eval()

    epoch = 70
    golden_model = copy.deepcopy(model_radius)
    best_loss = 10000000000000000000
    best_hinge_loss = 10000000000000000000
    for i in range(epoch):
        permutation = torch.randperm(train_input.shape[0])
        train_input = train_input[permutation]
        train_label = train_label[permutation]
        for j in range(train_input.shape[0]):
            data = train_input[i,:]
            label = train_label[i,:]
            center = model_center(data)
            # center = res[:5]
            encoded = radius_encoder(data)
            res: torch.FloatTensor = model_radius(encoded)
            # radius = res.reshape((5,5))
            radius = create_positive_definite_matrix(res)

            hinge_loss = torch.nn.functional.relu(torch.norm(radius@(center-label))-1+alpha)

            det = torch.linalg.det(radius)
            # if det.item()<0:
            #     regularization_loss = torch.log(torch.linalg.det((radius+torch.eye(5)*0.01)@(radius+torch.eye(5)*0.01).T))
            # else:
            regularization_loss = torch.log(det+1)
            loss = hinge_loss+gamma*regularization_loss

            if np.isnan(loss.item()):
                print("stop")

            optimizer_radius.zero_grad()
            loss.backward()
            optimizer_radius.step()

            # print(f"{i}, {j}: {loss.item()}, {hinge_loss.item()}, {regularization_loss.item()}")

        model_radius.eval()
        with torch.no_grad():
            res_center: torch.FloatTensor = model_center(train_input)
            encoded = radius_encoder(train_input)
            res_radius: torch.FloatTensor = model_radius(encoded)
            total_loss = 0
            total_hinge_loss = 0
            total_regularization_loss = 0
            for j in range(train_input.shape[0]):
                center = res_center[j,:]
                radius = create_positive_definite_matrix(res_radius[i,:])
                # radius = res_radius[j,:].reshape((5,5))
                label = train_label[i,:]

                hinge_loss = torch.nn.functional.relu(torch.norm(radius@(center-label))-1)
                regularization_loss = torch.log(torch.det(radius)+1)
                total_hinge_loss += hinge_loss 
                total_regularization_loss += regularization_loss
                loss = hinge_loss+gamma*regularization_loss
                total_loss += loss 

            print(f">>>>>> {i}: {total_loss.mean().item()}, {total_hinge_loss.item()}")
        if total_hinge_loss.item() < best_hinge_loss:
            if total_loss.mean().item()<best_loss:
                golden_model = copy.deepcopy(model_radius)
                best_loss = total_loss.mean().item()

        model_radius.train()

    res_center: torch.FloatTensor = model_center(train_input)
    encoded = radius_encoder(train_input)
    res_radius: torch.FloatTensor = model_radius(encoded)
    total_loss = 0
    total_hinge_loss = 0
    total_regularization_loss = 0

    for j in range(train_input.shape[0]):
        center = res_center[j,:]
        radius = create_positive_definite_matrix(res_radius[i,:])
        # radius = res_radius[j,:].reshape((5,5))
        label = train_label[i,:]

        hinge_loss = torch.nn.functional.relu(torch.norm(radius@(center-label))-1)
        regularization_loss = torch.log(torch.det(radius)+1)
        total_hinge_loss += hinge_loss 
        total_regularization_loss += regularization_loss
        loss = hinge_loss+gamma*regularization_loss
        total_loss += loss 

        # print(j, hinge_loss.item(), regularization_loss.item(), loss.item())

    res_center = model_center(train_input)
    encoded = radius_encoder(train_input)
    res_radius = golden_model(encoded)

    res_center = res_center.detach().numpy()
    # res_radius = res_radius.detach().numpy() 
    train_input = train_input.detach().numpy()
    train_label = train_label.detach().numpy()

    total_num = 0
    contain = 0
    for i in range(train_input.shape[0]):
        total_num += 1
        radius = create_positive_definite_matrix(res_radius[i,:]).detach().numpy()
        center = res_center[i,:]
        x = train_label[i,:]
        if np.linalg.norm(radius@radius.T@(center-x))<=1:
            contain += 1 
    print(contain, total_num, contain/total_num)

    plt.figure(0)
    plt.plot(train_input[:,0], train_label[:,0],'b*')
    plt.plot(train_input[:,0], res_center[:,0],'g*')

    plt.figure(1)
    plt.plot(train_input[:,1], train_label[:,1], 'b*')
    plt.plot(train_input[:,1], res_center[:,1],'g*')

    plt.figure(2)
    plt.plot(train_input[:,2], train_label[:,2], 'b*')
    plt.plot(train_input[:,2], res_center[:,2],'g*')

    plt.figure(3)
    plt.plot(train_input[:,3], train_label[:,3], 'b*')
    plt.plot(train_input[:,3], res_center[:,3],'g*')

    plt.figure(4)
    plt.plot(train_input[:,4], train_label[:,4], 'b*')
    plt.plot(train_input[:,4], res_center[:,4],'g*')

    plt.figure(5)
    ax = plt.gca()

    for i in range(train_input.shape[0]):
        # max_val = max_dimension_value(res_radius[i,:].reshape((5,5)), res_center[i,:], 0)
        # min_val = min_dimension_value(res_radius[i,:].reshape((5,5)), res_center[i,:], 0)
        mat = create_positive_definite_matrix(res_radius[i,:]).detach().numpy()
        # mat = mat@mat.T
        plt.figure(0)
        min_val, max_val = min_max_dimension_values(mat, res_center[i,:], 0)
        plt.plot([train_input[i,0],train_input[i,0]], [min_val,max_val], 'r*')

        plt.figure(1)
        min_val, max_val = min_max_dimension_values(mat, res_center[i,:], 1)
        plt.plot([train_input[i,1],train_input[i,1]], [min_val,max_val], 'r*')

        plt.figure(2)
        min_val, max_val = min_max_dimension_values(mat, res_center[i,:], 2)
        plt.plot([train_input[i,2],train_input[i,2]], [min_val,max_val], 'r*')

        plt.figure(3)
        min_val, max_val = min_max_dimension_values(mat, res_center[i,:], 3)
        plt.plot([train_input[i,3],train_input[i,3]], [min_val,max_val], 'r*')

        plt.figure(4)
        min_val, max_val = min_max_dimension_values(mat, res_center[i,:], 4)
        plt.plot([train_input[i,4],train_input[i,4]], [min_val,max_val], 'r*')

    for i in range(0, train_input.shape[0], 100):
        mat = create_positive_definite_matrix(res_radius[i,:]).detach().numpy()
        mat = mat[:2,:2]
        # mat = mat.T@mat
        plot_ellipsoid_2d(mat, res_center[i,:2], ax=ax)
        ax.plot(train_label[i,0],train_label[i,1],'b*')
        pass 
    # plt.plot(train_input[:,0], res_center[:,0]+,'r*')

    torch.save(model_center.state_dict(), 'pc_center.pth')
    torch.save(model_radius.state_dict(), 'pc_radius.pth')
    plt.show()