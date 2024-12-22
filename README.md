# **TrendScan**  

**TrendScan** is a web application designed to help users discover the latest fashion trends by aggregating data from multiple sources. It collects trending topics, styles, and products from Pinterest and popular e-commerce clothing sites to give users a real-time view of what's popular in the fashion world. Whether you're looking for inspiration for your next outfit or staying on top of the hottest trends, TrendScan provides the insights you need.

### **Features**  

- **Real-Time Trend Aggregation**: Scrapes and aggregates trending fashion topics from various sources like Pinterest and popular e-commerce sites.
- **Customizable Time Filters**: View the latest trends from the past day, week, or month.
- **Interactive Web App**: Explore trends via a sleek and user-friendly interface.
- **Search Functionality**: Search for specific fashion topics, styles, and trends.
- **Future Integration**: Potential integration with image recognition technology (myCloset) to connect personal wardrobes to trending styles.

### **Tech Stack**  

- **Frontend**:  
  - **Next.js**: Framework for server-side rendering and fast development of React-based apps.
  - **Tailwind CSS**: Utility-first CSS framework for fast, responsive design.

- **Backend**:  
  - **Node.js**: JavaScript runtime for handling API requests and server-side logic.
  - **Express.js**: Web framework for Node.js for API routing and handling.
  - **Supabase**: Backend-as-a-Service (BaaS) for real-time databases and authentication.

- **Data Sources**:  
  - **Pinterest API**: To collect trending fashion-related data.
  - **Web Scraping (via BeautifulSoup)**: For scraping additional fashion websites, blogs, and e-commerce stores.

- **Deployment**:  
  - **Vercel**: For hosting the frontend application.
  - **Supabase**: For user data and real-time trend data storage.

### **How to Get Started**

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/taliakusmirek/TrendScan.git
   cd trendscan
   ```

2. **Install dependencies**:  
   ```bash
   npm install
   ```

3. **Set up environment variables**:  
   - Create a `.env` file in the root of the project and add your API keys for Pinterest and Instagram.

   Example:  
   ```env
   PINTEREST_KEY=your_pinterest_api_key
   ```

4. **Run the development server**:  
   ```bash
   npm run dev
   ```

5. **Access the app**:  
   Navigate to `http://localhost:3000` in your browser.

### **Future Features**

- **Image Recognition Integration**: Connect with **myCloset** to provide personalized recommendations based on your closet’s content.
- **User Profiles**: Allow users to save their favorite trends, outfits, and fashion inspirations.
- **Outfit Recommendations**: Use aggregated trends to suggest full outfits based on the latest styles.

### **Contributing**

If you'd like to contribute to **TrendScan**, feel free to fork the repo and submit a pull request. Here's how you can contribute:

1. Fork the repository.
2. Create your branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -am 'Add new feature'`.
4. Push to your branch: `git push origin feature-name`.
5. Submit a pull request.

### **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
