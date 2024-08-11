import numpy as np
import matplotlib.pyplot as plt

def load_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=',')
    paths = []

    for identifier in np.unique(data[:, 0]):
        subset = data[data[:, 0] == identifier][:, 1:]
        segments = []

        for sub_id in np.unique(subset[:, 0]):
            segment = subset[subset[:, 0] == sub_id][:, 1:]
            if segment.shape[1] == 2:
                segments.append(segment)

        paths.append(segments)
    
    return paths

def visualize_paths(paths):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    palette = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    for idx, segments in enumerate(paths):
        color = palette[idx % len(palette)]
        for segment in segments:
            if segment.shape[1] == 2:
                ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=2)

    ax.set_aspect('equal')
    plt.show()

def find_lines(paths):
    line_segments = []
    for segments in paths:
        for segment in segments:
            if segment.shape[1] == 2:
                params = np.polyfit(segment[:, 0], segment[:, 1], 1)
                fit_line = np.polyval(params, segment[:, 0])
                if np.allclose(segment[:, 1], fit_line, atol=1e-2):
                    line_segments.append(segment)
    return line_segments

def find_circles(paths):
    circle_segments = []
    for segments in paths:
        for segment in segments:
            if segment.shape[1] == 2:
                x, y = segment[:, 0], segment[:, 1]
                centroid_x, centroid_y = np.mean(x), np.mean(y)
                radii = np.sqrt((x - centroid_x)**2 + (y - centroid_y)**2)
                if np.allclose(radii, np.mean(radii), atol=1e-2):
                    circle_segments.append(segment)
    return circle_segments

def check_symmetry(paths, axis='x'):
    symmetric_segments = []
    for segments in paths:
        for segment in segments:
            if segment.shape[1] == 2:
                if axis == 'x':
                    symmetric = np.allclose(segment[:, 0], -segment[:, 0][::-1])
                elif axis == 'y':
                    symmetric = np.allclose(segment[:, 1], -segment[:, 1][::-1])
                if symmetric:
                    symmetric_segments.append(segment)
    return symmetric_segments

def complete_curve(segment, method='linear'):
    if segment.shape[1] != 2:
        return segment

    if method == 'linear':
        x = segment[:, 0]
        y = segment[:, 1]
        filled_curve = np.column_stack([np.interp(np.linspace(x.min(), x.max(), len(x)), x, y)])
        return filled_curve

def execute():
    paths = load_csv('frag0.csv')
    visualize_paths(paths)

    lines = find_lines(paths)
    circles = find_circles(paths)
    visualize_paths([lines, circles])

    symmetric = check_symmetry(paths, axis='x')
    visualize_paths([symmetric])

    for segments in paths:
        for segment in segments:
            completed = complete_curve(segment)
            visualize_paths([[completed]])

if __name__ == "__main__":
    execute()
