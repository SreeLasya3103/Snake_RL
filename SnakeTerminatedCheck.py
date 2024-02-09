## Checks if snake has collided with the boundary of the grid
def boundary_collision(snake_head):
     # When snake dimensions go beyond the grid dimensions
    if snake_head[0]>=150 or snake_head[0]<0 or snake_head[1]>=150 or snake_head[1]<0 :
        return 1 # True
    else:
        return 0 # False

    
## Checks if snake has collided with it's own body
def self_collision(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0

## Maximum length of snake before the episode terminates
MAX_SNAKE_LEN = 15
