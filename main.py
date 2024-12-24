from financial_personality_assistant import FinancialPersonalityAssistant

def validate_transaction(transaction):
    if not isinstance(transaction, list) or len(transaction) != 3:
        return False
    if not isinstance(transaction[0], str) or not isinstance(transaction[1], str):
        return False
    try:
        float(transaction[2])
        return True
    except (ValueError, TypeError):
        return False

def get_sample_transactions():
    # In a real application, this would come from user's bank feed or manual input
    return [
        ["groceries", "walmart", 18.00],
        ["investment", "stocks", 1000000.00],
        ["shopping", "amazon", 566.50],
        ["utilities", "electric", 110.00],
        ["savings", "transfer", 540.00]
    ]

def main():
    try:
        print("Welcome to Financial Personality Assistant!")
        print("Analyzing your transactions...\n")

        assistant = FinancialPersonalityAssistant()
        transactions = get_sample_transactions()
        
        # Validate transactions
        valid_transactions = [t for t in transactions if validate_transaction(t)]
        if not valid_transactions:
            raise ValueError("No valid transactions found")
            
        # Process the data
        result = assistant.process_user_data(valid_transactions)
        
        # Display results
        print(f"Based on your spending patterns, your financial personality type is: {result['personality']}")
        print("\nHere are some personalized recommendations for you:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec}")
            
        print(f"\nYour Financial Health Score: {result['user_rating']}/10")
        print("\nScore interpretation:")
        print("9-10: Excellent financial management")
        print("7-8.9: Good financial habits")
        print("5-6.9: Room for improvement")
        print("Below 5: Needs significant attention")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return

if __name__ == "__main__":
    main()
