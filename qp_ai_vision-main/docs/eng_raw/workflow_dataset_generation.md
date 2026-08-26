# Dataset Generation

## 1. Purpose

To study the learning ability of artificial intelligence models for different visual features, a controllable dataset generator is required.

By controlling the basic variables in the dataset, we can:

* Analyze model sensitivity to specific features
* Study model generalization ability
* Verify whether the model has learned the intended features
* Automatically generate large amounts of training data
* Construct a standardized experimental environment

---

## 2. Core Variables

### 2.1 Geometric Variables

Controls the geometric properties of the target object.

| Variable   | Description          |
| ---------- | -------------------- |
| Radius     | Circle radius (size) |
| Width      | Object width         |
| Height     | Object height        |
| Position X | Horizontal position  |
| Position Y | Vertical position    |
| Rotation   | Rotation angle       |
| Scale      | Scaling factor       |
| Thickness  | Border thickness     |

Example:

```python
radius = 50
position = (250, 250)
thickness = 3
```

---

### 2.2 Color Variables

Controls the colors of the target object and the background.

| Variable         | Description                |
| ---------------- | -------------------------- |
| Object Color     | Color of the target object |
| Border Color     | Color of the object border |
| Background Color | Background color           |
| Brightness       | Image brightness level     |
| Contrast         | Image contrast level       |

Example:

```python
object_color = (255, 0, 0)
background_color = (255, 255, 255)
```

---

### 2.3 Shape Variables

Controls the category and structural characteristics of target objects.

| Variable          | Description                       |
| ----------------- | --------------------------------- |
| Shape Type        | Object shape category             |
| Number of Objects | Number of target objects          |
| Shape Complexity  | Geometric complexity of the shape |

Example shapes:

* Circle
* Triangle
* Square
* Rectangle
* Polygon
* Custom Shape

---

### 2.4 Image Quality Variables

Controls image quality and environmental effects.

| Variable            | Description               |
| ------------------- | ------------------------- |
| Noise Level         | Amount of image noise     |
| Blur Level          | Degree of image blur      |
| Brightness          | Overall image brightness  |
| Contrast            | Overall image contrast    |
| Compression Quality | Image compression quality |

Example:

```python
noise_level = 0.1
blur_level = 3
```

---

## 3. Dataset Generation Strategies

### 3.1 Fixed Variable Experiment

Only one variable is changed while all other variables remain fixed.

Purpose:

* Study the model's sensitivity to a specific factor.
* Evaluate the influence of a single variable on model performance.

Example:

```text
Radius:
20
40
60
80
100
```

Fixed parameters:

```text
Object Color = Black
Background Color = White
Position = Center
```

---

### 3.2 Combination Experiment

Multiple variables are combined to generate different samples.

Example:

```python
radius = [20, 50, 100]
color = ["red", "green", "blue"]
```

Generated combinations:

```text
20-red
20-green
20-blue

50-red
50-green
50-blue

100-red
100-green
100-blue
```

Total combinations:

```text
3 × 3 = 9
```

---

### 3.3 Random Generation

Variables are randomly generated within predefined ranges.

Example:

```python
radius = random.randint(20, 100)

object_color = (
    random.randint(0, 255),
    random.randint(0, 255),
    random.randint(0, 255)
)
```

Advantages:

* Unlimited dataset size
* Reduced risk of overfitting

Disadvantages:

* Harder to analyze the effect of individual variables

---

### 3.4 Domain Randomization

All configurable variables are randomized.

Randomized properties may include:

* Size
* Color
* Position
* Rotation
* Noise
* Blur
* Background
* Lighting conditions

Example:

```python
radius = random.randint(20, 100)

object_color = random_rgb()

background_color = random_rgb()

position = random_position()

noise = random_noise()
```

Purpose:

* Improve model robustness
* Increase generalization to real-world environments

---


## 4. Dataset Configuration

Recommended configuration structure:

```python
DatasetConfig(
    image_width=500,
    image_height=500,

    radius_min=20,
    radius_max=100,

    object_color_range=None,
    background_color_range=None,

    noise_level=0.0,
    blur_level=0,

    sample_count=1000
)
```

Benefits:

* Centralized parameter management
* Reproducible experiments
* Easy extension and maintenance

---

## 5. Dataset Implementation

Metadata generation workflow
```
create_metadata()
        │
        ▼
metadata
        │
        ▼
write_image_metadata(...)
        │
        ▼
metadata
        │
        ├──────────────┐
        ▼              ▼
create_circle_metadata(...)
create_triangle_metadata(...)
        │              │
        ▼              ▼
add_object_metadata(...)
add_object_metadata(...)
        │
        ▼
metadata
        │
        ▼
write_metadata_file(metadata)

```

## 6. Future Extensions

Current support:

```text
Circle Dataset
```

Potential future extensions:

```text
Rectangle Dataset
Triangle Dataset
Polygon Dataset

Classification Dataset
Object Detection Dataset
Segmentation Dataset
Pose Estimation Dataset
```

All dataset types should share the same configuration and generation framework.

---

## 7. Recommended Research Variables

The following variables are particularly useful for machine learning experiments:

| Priority | Variable            |
| -------- | ------------------- |
| High     | Size                |
| High     | Position            |
| High     | Rotation            |
| High     | Object Color        |
| High     | Background Color    |
| Medium   | Noise               |
| Medium   | Blur                |
| Medium   | Brightness          |
| Medium   | Contrast            |
| Low      | Compression Quality |

These variables help evaluate model robustness against geometric changes, color variations, and image quality degradation.

---

## 8. Conclusion

A dataset generator should be designed as a configurable, extensible, and reproducible platform.

Main objectives:

1. Control dataset variables
2. Automatically generate training samples
3. Support systematic experimentation
4. Enable future task extensions
5. Improve model generalization

By using a unified configuration system, different types of computer vision datasets can be generated efficiently for AI research and development.
