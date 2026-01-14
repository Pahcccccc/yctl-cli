#!/bin/bash

# yctl - Example Usage Script
# This script demonstrates all the commands available in yctl

echo "=========================================="
echo "yctl - Personal AI Engineer CLI Tool"
echo "Example Usage Demonstration"
echo "=========================================="
echo ""

# 1. Check version
echo "1. Checking yctl version..."
yctl --version
echo ""

# 2. Run system health check
echo "2. Running system health check..."
yctl doctor
echo ""
read -p "Press Enter to continue..."
echo ""

# 3. Analyze an AI idea
echo "3. Analyzing an AI project idea..."
yctl think "sentiment analysis for customer reviews"
echo ""
read -p "Press Enter to continue..."
echo ""

# 4. Create a sample NLP project
echo "4. Creating a sample NLP project..."
yctl init nlp demo-sentiment-analyzer
echo ""
read -p "Press Enter to continue..."
echo ""

# 5. Create a sample dataset for inspection
echo "5. Creating a sample dataset..."
cat > /tmp/sample_data.csv << EOF
id,text,sentiment,rating
1,"Great product! Love it",positive,5
2,"Terrible experience",negative,1
3,"It's okay, nothing special",neutral,3
4,"Amazing quality",positive,5
5,"Worst purchase ever",negative,1
6,"Good value for money",positive,4
7,"Not worth it",negative,2
8,"Decent product",neutral,3
9,"Highly recommend!",positive,5
10,"Disappointed",negative,2
EOF

echo "Sample dataset created at /tmp/sample_data.csv"
echo ""

# 6. Inspect the dataset
echo "6. Inspecting the sample dataset..."
yctl inspect /tmp/sample_data.csv
echo ""
read -p "Press Enter to continue..."
echo ""

# 7. Show help
echo "7. Showing help for all commands..."
echo ""
echo "=== Main Help ==="
yctl --help
echo ""

echo "=== Init Command Help ==="
yctl init --help
echo ""

echo "=== Inspect Command Help ==="
yctl inspect --help
echo ""

echo "=== Doctor Command Help ==="
yctl doctor --help
echo ""

echo "=== Think Command Help ==="
yctl think --help
echo ""

echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. cd demo-sentiment-analyzer"
echo "  2. source venv/bin/activate"
echo "  3. pip install -r requirements.txt"
echo "  4. Start building your AI project!"
echo ""
