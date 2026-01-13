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
## 🏗 System Architecture (Mermaid Diagram)

## 🏗 System Architecture (Mermaid Diagram)

```mermaid
flowchart TD
    A[Corporate Login System] --> B[Admin Module]
    
    B --> B1[PFEP Upload & Standardization]
    B --> B2[Master Data Locking]
    B --> B3[Tolerance & Ideal Days Settings]

    B --> C[User Module]

    C --> C1[Inventory Upload]
    C --> C2[Matching & Validation]

    C --> D[Inventory Analysis Engine]

    D --> D1[Excess/Shortage Detection]
    D --> D2[Deviation Value Calculation]
    D --> D3[Vendor-wise Summary]
    D --> D4[KPI Computation]

    D --> E[Dashboards & Visualizations]

    E --> E1[Plotly Charts]
    E --> E2[Top 10 Parts Analysis]
    E --> E3[Norm vs Actual Comparison]

    E --> F[PPT Report Generator]
    
    F --> F1[Cover Slide]
    F --> F2[KPI Performance Slide]
    F --> F3[Status Breakdown Slide]
