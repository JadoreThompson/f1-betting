# **Overview**
An F1 betting platform with an odd generation model. 

# **Categories**
To reduce the number of classes the model needs to predict, I implemented multiple condensed representations of driver finishing positions. 

### **Tight**
Designed for fine-grained performance differences at the top end of the field.

| Position | Category |
| -------- | -------- |
| 1st      | `"1"`    |
| 2nd      | `"2"`    |
| 3rd      | `"3"`    |
| 4th–5th  | `"4"`    |
| 6th–10th | `"5"`    |
| >10th    | `"6"`    |


---
### **Loose**
Designed for broad classifications and for models that struggle with overfitting.

| Position | Category |
| -------- | -------- |
| 1st–3rd  | `"1"`    |
| 4th–5th  | `"2"`    |
| 6th–10th | `"3"`    |
| >10th    | `"4"`    |


---
### **Binary**
A win/loss binary classifier:

| Position | Category |
| -------- | -------- |
| 1st      | `"win"`  |
| 2nd+     | `"lose"` |


# **Models**
#### <u>loss_1</u>
This is the first model implemented for the loose classification mentioned prior. Implemented with a random forest, this model achieves a 50% success rate on the 2024 season on which it was evaluated on.

Features used:
- `grid` (*int*): The starting grid position.
- `position_quali` (*int*): The final position achieved after Q3.
- `sma_position` (*float*): The simple moving average of the last 5 races. 
- `avg_pos_move` (*float*): The average amount of positions the driver moves from their starting grid position.
# **Requirements**
- Python 3.12
