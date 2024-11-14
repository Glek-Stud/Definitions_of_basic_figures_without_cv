from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


def sobel_edge_detection(binary_matrix):
    Gx = np.array([[-1, 0, 1], 
                   [-2, 0, 2], 
                   [-1, 0, 1]])
    
    Gy = np.array([[-1, -2, -1], 
                   [0,  0,  0], 
                   [1,  2,  1]])

    padded_matrix = np.pad(binary_matrix, pad_width=1, mode='constant', constant_values=0)

    rows, cols = padded_matrix.shape

    edge_magnitude = np.zeros((rows - 2, cols - 2), dtype=int)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = padded_matrix[i - 1:i + 2, j - 1:j + 2]
            
            gx = np.sum(Gx * region)
            gy = np.sum(Gy * region)
            
            edge_magnitude[i - 1, j - 1] = abs(gx) + abs(gy)

    unique_magnitudes = np.unique(edge_magnitude)
    print("Величини країв:", unique_magnitudes)

    edge_threshold = 4 
    edges = (edge_magnitude >= edge_threshold).astype(int)
    
    return edges



def find_bounding_boxes(binary_matrix):
    rows, cols = len(binary_matrix), len(binary_matrix[0])
    visited = [[False] * cols for _ in range(rows)]
    bounding_boxes = []

    def bfs(x, y):
        min_x, max_x, min_y, max_y = x, x, y, y
        queue = [(x, y)]
        visited[x][y] = True
        
        while queue:
            cx, cy = queue.pop(0)
            min_x = min(min_x, cx)
            max_x = max(max_x, cx)
            min_y = min(min_y, cy)
            max_y = max(max_y, cy)
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < rows and 0 <= ny < cols and binary_matrix[nx][ny] == 1 and not visited[nx][ny]:
                    visited[nx][ny] = True
                    queue.append((nx, ny))
        
        return (min_x, min_y, max_x, max_y)

    for i in range(rows):
        for j in range(cols):
            if binary_matrix[i][j] == 1 and not visited[i][j]:
                bounding_boxes.append(bfs(i, j))

    return bounding_boxes



def calculate_max_radius(bounding_boxes):
    radius_range = []
    
    for box in bounding_boxes:
        min_x, min_y, max_x, max_y = box
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        max_radius = min(width, height)
        radius_range.append(max_radius)

        randge_min = min(radius_range) // 2
        if randge_min <= 5:
            randge_min = 5
        
        randge_max = max(radius_range)
        radius_range_x2 = range(randge_min, randge_max)

    return radius_range_x2



def hough_circle_transform(edges, radius_range, threshold_factor, min_distance, theta_step):
    rows, cols = edges.shape
    accumulator = np.zeros((rows, cols, len(radius_range)), dtype=int)

    for x in range(rows):
        for y in range(cols):
            if edges[x, y] == 1:
                for r_idx, radius in enumerate(radius_range):
                    for theta in range(0, 360, theta_step):  
                        a = int(x - radius * np.cos(np.radians(theta)))
                        b = int(y - radius * np.sin(np.radians(theta)))
                        if 0 <= a < rows and 0 <= b < cols:
                            accumulator[a, b, r_idx] += 1

    detected_circles = []
    max_votes = np.max(accumulator)
    threshold = max_votes * threshold_factor

    for r_idx, radius in enumerate(radius_range):
        for a in range(1, rows - 1):
            for b in range(1, cols - 1):
                if (accumulator[a, b, r_idx] > threshold and
                    accumulator[a, b, r_idx] > accumulator[a-1, b, r_idx] and
                    accumulator[a, b, r_idx] > accumulator[a+1, b, r_idx] and
                    accumulator[a, b, r_idx] > accumulator[a, b-1, r_idx] and
                    accumulator[a, b, r_idx] > accumulator[a, b+1, r_idx]):
                    
                    detected_circles.append((a, b, radius, accumulator[a, b, r_idx]))

    detected_circles = sorted(detected_circles, key=lambda x: x[3], reverse=True)
    
    final_circles = []
    for circle in detected_circles:
        if all(np.sqrt((circle[0] - c[0])**2 + (circle[1] - c[1])**2) > min_distance for c in final_circles):
            final_circles.append(circle[:3])
    
    return final_circles


input_path = 'input/Test.bmp'
img = Image.open(input_path).convert('1')  
binary_matrix = np.array(img, dtype=int)
print("Матриця:\n", binary_matrix)

edges = sobel_edge_detection(binary_matrix)
print("Границі:\n", edges)

box = find_bounding_boxes(binary_matrix)
print("Периметри:\n", box)

radius_range = calculate_max_radius(box)
print("Границя радіусів:\n", radius_range)

#threshold_factor = 0.8 #Test2
threshold_factor = 0.9 #Test
#threshold_factor = 0.7 #Test3
#threshold_factor = 0.8 #Test5

min_distance = 10
theta_step = 2

circles = hough_circle_transform(edges, radius_range, threshold_factor, min_distance, theta_step)
diameters = [(x, y, radius * 2) for (x, y, radius) in circles]

for (x, y, diameter) in diameters:
    print(f"Коло з центром в ({x}, {y}), його діаметр: {diameter}")

original_img = Image.open(input_path).convert('RGB')
draw = ImageDraw.Draw(original_img)
for circle in circles:
    x, y, radius = circle
    draw.ellipse((y - radius, x - radius, y + radius, x + radius), outline="red", width=1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(binary_matrix, cmap='gray')
plt.title('Бінарне зображення')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_matrix, cmap='gray')
plt.imshow(edges, cmap='gray') 
plt.title('Границі обектів')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(original_img)
plt.title('Знайдені кола')
plt.axis('off')

plt.tight_layout()
plt.show()

