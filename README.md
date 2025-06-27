This project is a Streamlit-based recommendation system that suggests similar products to online retail customers based on product descriptions. It uses TF-IDF text vectorization and cosine similarity to find and recommend similar items.

🚀 Features
✅ Simple, interactive Streamlit web app
✅ TF-IDF-based product similarity
✅ Top 5 product recommendations
✅ Shows top 5 most sold products
✅ CSS-styled UI for better appearance
✅ Sample product names for user guidance

📂 Project Structure

├── clean_100_products.csv      # cleaned dataset of top 100 products
├── online_recommend.py         # Streamlit app
├── README.md


⚙️ Tech Stack
Python
Streamlit
Pandas
Scikit-learn
CSS (inline for Streamlit)


💡 How to Run
Clone this repository:
git clone https://github.com/yourusername/your-repo-name.git

Install dependencies:
pip install -r requirements.txt

Launch the Streamlit app:
streamlit run online_recommend.py

📊 Dataset
The project uses a cleaned subset of 100 products from the original Online Retail dataset.
The file clean_100_products.csv is included for easy running.

📝 Usage
Enter a product name exactly (case-sensitive).
Get top 5 similar recommendations based on product description.
New users can explore the top 5 most sold products shown on the homepage.

🙌 Contributing
Pull requests welcome! Please open an issue first if you want to discuss changes.

📧 Contact
If you have any questions, reach out to:
[lalitsinghbisht282002@gmail.com]

