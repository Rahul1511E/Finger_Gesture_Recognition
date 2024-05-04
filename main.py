import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def count_fingers(hand_landmarks):
    tip_ids = [8, 12, 16, 20]  # Finger landmark ids for the tips
    arr = []
    if(hand_landmarks.landmark[4].y < hand_landmarks.landmark[5].y):
        arr.append(1)
    else:
        arr.append(0)
    for id in tip_ids:
        if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y:
            arr.append(1)
        else:
            arr.append(0)
    count = 0
    if arr == [1,0,0,0,0]:
        count = 6
    elif arr == [0,1,0,0,0] or arr == [0,0,1,0,0] or arr == [0,0,0,1,0] or arr == [0,0,0,0,1]:
        count = 1
    elif arr == [0,1,1,0,0] or arr == [0,0,1,1,0] or arr == [0,0,0,1,1]:
        count = 2
    elif arr == [0,1,1,1,0] or arr == [0,0,1,1,1]:
        count = 3
    elif arr == [0,1,1,1,1]:
        count = 4
    elif arr == [1,1,1,1,1]:
        count = 5
    elif arr == [1,1,0,0,0] or arr == [1,0,1,0,0] or arr == [1,0,0,1,0]:
        count = 7
    elif arr == [1,1,1,0,0] or arr == [1,0,1,1,0]:
        count = 8
    elif arr == [1,1,1,1,0] or arr == [1,0,1,1,1]:
        count = 9
    elif arr == [1,0,0,0,1]:
        count = 10
    else:
        count = 0
    return count

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hand_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not hand_detected:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_detected = True
                    break

        if hand_detected:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_d = mp.solutions.drawing_utils
                mp_d.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cx, cy = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * small_frame.shape[1]), \
                    int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * small_frame.shape[0])

                height, width, _ = frame.shape

                num_fingers = count_fingers(hand_landmarks)

                if num_fingers == 1:
                    cv2.putText(frame, "Emergency", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 230), 2)
                elif num_fingers == 2:
                    cv2.putText(frame, "Minor Complications", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 83, 243), 2)
                elif num_fingers == 3:
                    cv2.putText(frame, "Medicines", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 83, 243), 2)
                elif num_fingers == 4:
                    cv2.putText(frame, "Food or Water", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 83, 243), 2)
                elif num_fingers == 5:
                    cv2.putText(frame, "No Requirements", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                elif num_fingers == 6:
                    cv2.putText(frame, "Speak Assistance", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (242, 157, 10), 2)
                elif num_fingers == 7:
                    cv2.putText(frame, "Personal Needs", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 242, 10), 2)
                elif num_fingers == 8:
                    cv2.putText(frame, "Restroom", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (242, 157, 10), 2)
                elif num_fingers == 9:
                    cv2.putText(frame, "Bathing or Clothing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (10, 242, 10), 2)
                elif num_fingers == 10:
                    cv2.putText(frame, "Sleeping Arrangements", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (242, 157, 10), 2)
                else:
                    cv2.putText(frame, "No Requirements", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()