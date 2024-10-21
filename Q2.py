import cv2
import mediapipe as mp
import numpy as np
# Initialize MediaPipe Hands
mphands=mp.solutions.hands
hands = mphands.Hands()

# Game settings
width, height = 1280, 640
player_pos = [0, 0]
# enemy speed, size, and list initialization
blocks = []
enemy_speed = 5
enemy_size = 50

# Initialize score
score = 0

# Create random enemy
def create_enemy():
    x_pos = np.random.randint(0, width - enemy_size)
    return [x_pos, 0]
# Move enemies down
def move_enemies(blocks):
    for enemy in blocks:
        enemy[1] += enemy_speed
   
# Check if enemy is off-screen
def check_off_screen(blocks):
    global score
    for enemy in blocks[:] :
        if enemy[1] > height:
            score = score+1
            blocks.remove(enemy)
# Increment score for each enemy that goes off-screen
# Check for collisions
def check_collision(player_pos, blocks):
    pos_x, pos_y = player_pos
    for enemy in blocks:
        if (pos_x < enemy[0] + enemy_size and pos_x + enemy_size > enemy[0] and pos_y < enemy[1] + enemy_size and pos_y + enemy_size > enemy[1]):
            return True
    return False
        
   
# Initialize webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    result = hands.process(rgb_frame)

            
    # Get coordinates of the index finger tip (landmark 8)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the coordinates of specific landmarks (e.g., the tip of the index finger)
            index_finger_tip = hand_landmarks.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]
            player_pos[0] = int(index_finger_tip.x * width) 
            player_pos[1] = int(index_finger_tip.y * height)
    
    # Move player based on hand movement
    cv2.rectangle(frame,(player_pos[0],player_pos[1]),(player_pos[0]+50,player_pos[1]+50),(0,255,0),-1)

    # Add new enemies
    if np.random.randint(1, 20) == 10:
        blocks.append(create_enemy())
    
    # Move enemies
    move_enemies(blocks)
    
    # Check for collision
    if check_collision(player_pos,blocks)==True:
        print("Game Over")
        break
    
    # Draw game elements
    for enemy in blocks:
        cv2.rectangle(frame,(enemy[0],enemy[1]),(enemy[0]+enemy_size,enemy[1]+enemy_size),(0,0,255),-1)
    
    # Display score on the frame
    cv2.putText(frame, f'Score: {score}' , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Object Dodging Game", frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()