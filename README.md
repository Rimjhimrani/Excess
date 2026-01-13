# 📦 **AI-Driven Inventory Management System**

### *A Complete PFEP-Based Inventory Analytics Platform (Streamlit + Python)*

---

## 🚀 **Overview**

This project is a full-stack **Inventory Analysis Platform** designed for manufacturing and supply chain companies.
It allows users to upload their **PFEP master data** and daily **inventory dumps**, compares both datasets, and automatically generates:

✔ Inventory deviation analysis
✔ Excess & shortage detection
✔ Vendor-wise analytics
✔ Trend charts and dashboards
✔ A fully automated **PowerPoint Report**

The system includes **Admin/User roles**, **OTP-based password reset**, and **company-wise data isolation**, making it deployable for corporate use.

---

## 🧠 **Key Features**

### 🔐 **Corporate Login System**

* Admin/User role separation
* First-time login password setup
* OTP-based password recovery
* Company-wise data segregation

---

### 📁 **PFEP Master Upload (Admin)**

* Upload Excel/CSV PFEP files
* Intelligent column recognition (50+ variants)
* Auto-standardization of:

  * Part numbers
  * RM norms
  * Unit price
  * Vendor data
  * Consumption/day
* Server-side locking for user analysis
* Persistent storage using company IDs

---

### 📦 **Inventory Upload (User)**

* Upload daily/weekly inventory dump
* Auto-map PFEP vs Inventory
* Detect missing/excess/short parts
* Full part-level analysis

---

### 📊 **Analytics & Visualization**

Built using **Plotly**:

* Excess vs shortage bars
* Vendor-wise deviation charts
* Top 10 risky parts
* Norm vs Actual value comparison
* KPI dashboard
* Inventory days calculation
* Overall deviation in INR (Lakhs/Crores)

---

### 📝 **Automated PowerPoint Report**

Using `python-pptx`, the system generates a complete PPT with:

* Professional cover slide
* Inventory performance overview
* Status breakdown slide
* Auto-populated KPIs
* Vendor and part-level analysis
* Logo and branding placement
* Date & reference stamping

This makes it presentation-ready for CXOs and management reviews.

---

## 🛠 **Tech Stack**

| Category        | Technology                                |
| --------------- | ----------------------------------------- |
| Frontend        | Streamlit                                 |
| Backend         | Python                                    |
| Data Processing | Pandas, NumPy                             |
| Visualizations  | Plotly, GraphObjects                      |
| Reporting       | python-pptx                               |
| Security        | Pickle-based encrypted storage, OTP Email |
| Email           | SMTP (Gmail App Password)                 |
| Logging         | Python `logging` module                   |

---

## 📐 **System Architecture**

```
+-----------------------+
|  Corporate Login      |
+-----------------------+
        |
        v
+-----------------------+
|    Admin Dashboard    |
+-----------------------+
| Upload PFEP           |
| Set Tolerance         |
| Lock Data             |
+-----------------------+
        |
        v
+-----------------------+
|     User Module       |
+-----------------------+
| Upload Inventory      |
| Run Analysis          |
+-----------------------+
        |
        v
+-------------------------------+
| Inventory Analysis Engine     |
+-------------------------------+
| Excess / Short Detection      |
| Deviation Calculation         |
| Vendor Summary                |
| KPI Computation              |
+-------------------------------+
        |
        v
+-------------------------------+
| Interactive Dashboard         |
+-------------------------------+
| Plotly Charts                 |
| Top 10 Issues                |
| Norm vs Actual Comparison     |
+-------------------------------+
        |
        v
+-------------------------------+
| PPT Report Generator          |
+-------------------------------+
```

---

## 🚩 **Project Highlights (For Recruiter Attention)**

* Built as a **production-grade corporate tool**, not a student demo.
* Handles **real-world complexities** like mismatched columns, missing data, consumption calculations, tolerance zones, etc.
* Includes a full **authentication + security** layer.
* Automated end-to-end workflow from data upload to PPT export.
* Deployable on **Streamlit Cloud / AWS / GCP / Local servers**.

---

## 🧪 **Sample PFEP & Inventory Data**

The system contains demo datasets for quick testing:

* Sample PFEP master
* Sample inventory sheet
* Preloaded mappings
* Predefined unit prices and consumption values

---

## 📥 **Installation**

Clone the repository:

```bash
git clone https://github.com/yourusername/inventory-management-system.git
cd inventory-management-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit app:

```bash
streamlit run app.py
```

---

## 📎 **Project Screenshots**

(Add your own later)

```
📌 Dashboard preview  
📌 Vendor analysis graph  
📌 PPT report sample  
📌 Login module  
```

---

## 👤 **Developer**

**Rimjhim Rani**
Python Developer | Data Analytics | Automation

---

## ⭐ **How I Achieved This as a Fresher**

* Learned Streamlit for front-end UI
* Built complete authentication + OTP flow
* Mastered Pandas for large PFEP & inventory datasets
* Developed a custom rule engine for stock deviation
* Built PPT automation to replace manual reporting
* Implemented secure company-level data handling

---

## 💼 **Ideal Use Cases**

* Manufacturing companies
* Automotive suppliers
* Warehouse stock management
* PFEP & MRP teams
* Demand planning departments

---

## 🎯 **Future Enhancements**

* PostgreSQL / MongoDB backend
* Vendor email alerts
* Machine learning-based consumption forecasting
* Role-based dashboards
* API integration for ERP (SAP / Tally / Oracle)

