# Caroline Katko, Transy U
# Code that allows for human playing of blackjack
# written with the help of ChatGPT
# using the same OpenAI gym environment


#imports
import gymnasium as gym
import pygame

def human_play_blackjack():
    # Create the Blackjack environment with render_mode='human'
    env = gym.make('Blackjack-v1', render_mode='human')

    # Reset the environment to start the game
    state, _ = env.reset()
    print("Welcome to Blackjack!")
    print("Actions: 0 = Stand, 1 = Hit")

    def describe_state(state):
        player_sum, dealer_card, usable_ace = state
        ace_status = "Yes" if usable_ace else "No"
        return f"Your total: {player_sum}, Dealer's face-up card: {dealer_card}, Usable Ace: {ace_status}"

    running = True
    while running:
        # Ensure the Pygame event loop is processed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        print(f"\n{describe_state(state)}")

        # Ask the human player for an action
        action = None
        while action not in [0, 1]:
            try:
                action = int(input("Choose your action (0 = Stand, 1 = Hit): "))
            except ValueError:
                print("Invalid input. Please enter 0 or 1.")

        # Step through the environment
        state, reward, terminated, truncated, _ = env.step(action)

        # Check if the game is over
        if terminated or truncated:
            print("\nGame Over!")
            print(describe_state(state))  # Final state at game end

            # Extract dealer's full hand
            dealer_hand = env.unwrapped.dealer
            print(f"Dealer's full hand: {dealer_hand}")

            if reward > 0:
                print("You win!")
            elif reward < 0:
                print("You lose!")
            else:
                print("It's a draw!")

            # Ask if the player wants to play again
            play_again = input("Do you want to play again? (y/n): ").lower()
            if play_again == 'y':
                state, _ = env.reset()
            else:
                print("Thanks for playing!")
                running = False

    # Close the environment
    env.close()
    pygame.quit()

# Main function
if __name__ == "__main__":
    human_play_blackjack()
