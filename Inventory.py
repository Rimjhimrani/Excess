import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import io
import re
from typing import Any
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Inventory Management System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.graph-description {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 20px;
    font-style: italic;
    border-left: 4px solid #1f77b4;
}
.metric-container {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.status-excess {
    background-color: #ffebee;
    border-left: 4px solid #f44336;
}
.status-short {
    background-color: #fff3e0;
    border-left: 4px solid #ff9800;
}
.status-normal {
    background-color: #e8f5e8;
    border-left: 4px solid #4caf50;
}
.lock-button {
    background-color: #28a745;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    border: none;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CLASS: DataPersistence
# -----------------------------------------------------------------------------
class DataPersistence:
    """Handle data persistence across sessions"""
    
    @staticmethod
    def save_data_to_session_state(key, data):
        st.session_state[key] = {
            'data': data,
            'timestamp': datetime.now(),
            'saved': True
        }
    
    @staticmethod
    def load_data_from_session_state(key):
        if key in st.session_state and isinstance(st.session_state[key], dict):
            return st.session_state[key].get('data')
        return None
    
    @staticmethod
    def get_data_timestamp(key):
        if key in st.session_state and isinstance(st.session_state[key], dict):
            return st.session_state[key].get('timestamp')
        return None

# -----------------------------------------------------------------------------
# CLASS: InventoryAnalyzer
# -----------------------------------------------------------------------------
class InventoryAnalyzer:
    """Enhanced inventory analysis with comprehensive reporting"""
    
    def __init__(self):
        self.status_colors = {
            'Within Norms': '#4CAF50',    # Green
            'Excess Inventory': '#2196F3', # Blue
            'Short Inventory': '#F44336'   # Red
        }
        
    def safe_float_convert(self, value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def analyze_inventory(self, pfep_data, current_inventory, tolerance=None):
        if tolerance is None:
            tolerance = st.session_state.get("admin_tolerance", 30)
            
        results = []
        pfep_dict = {str(item['Part_No']).strip().upper(): item for item in pfep_data}
        inventory_dict = {str(item['Part_No']).strip().upper(): item for item in current_inventory}
        
        for part_no, inventory_item in inventory_dict.items():
            pfep_item = pfep_dict.get(part_no)
            if not pfep_item:
                continue
            
            try:
                current_qty = float(inventory_item.get('Current_QTY', 0)) or 0.0
                part_desc = pfep_item.get('Description', '')
                unit_price = float(pfep_item.get('unit_price', 0)) or 1.0
                rm_qty = self.safe_float_convert(pfep_item.get('RM_IN_QTY', 0))
                
                current_value = current_qty * unit_price
                lower_bound = rm_qty * (1 - tolerance / 100)
                upper_bound = rm_qty * (1 + tolerance / 100)
                revised_norm_qty = upper_bound

                deviation_qty = current_qty - revised_norm_qty
                deviation_value = deviation_qty * unit_price

                if current_qty < lower_bound:
                    status = 'Short Inventory'
                elif current_qty > upper_bound:
                    status = 'Excess Inventory'
                else:
                    status = 'Within Norms'

                result = {
                    'PART NO': part_no,
                    'PART DESCRIPTION': part_desc,
                    'Vendor Name': pfep_item.get('Vendor_Name', 'Unknown'),
                    'Vendor_Code': pfep_item.get('Vendor_Code', ''),
                    'RM Norm - In Qty': rm_qty,
                    'Revised Norm Qty': revised_norm_qty,
                    'Lower Bound Qty': lower_bound,
                    'Upper Bound Qty': upper_bound,
                    'UNIT PRICE': unit_price,
                    'Current Inventory - Qty': current_qty,
                    'Current Inventory - VALUE': current_value,
                    'Stock Deviation Value': deviation_value,
                    'Status': status,
                    'INVENTORY REMARK STATUS': status
                }
                results.append(result)
            except Exception:
                continue
        return results 
        
    def show_vendor_chart_by_status(self, processed_data, status_filter, chart_title, chart_key, color, unit_type='Lakhs', top_n=10):
        """
        Show top N vendors by deviation value with dynamic Unit selection.
        """
        # Define divisor and suffix based on unit_type
        if unit_type == 'Millions':
            divisor = 1_000_000
            suffix = "M"
            y_axis_label = f"Value (Millions)"
        else:  # Default to Lakhs
            divisor = 100_000
            suffix = "L"
            y_axis_label = f"Value (Lakhs)"

        # Filter by inventory status
        filtered = [item for item in processed_data if item.get('INVENTORY REMARK STATUS') == status_filter]
        vendor_totals = {}
        vendor_counts = {}
        
        for item in filtered:
            vendor = item.get('Vendor Name', 'Unknown')
            try:
                current_qty = float(item.get('Current Inventory - Qty', 0) or 0)
                norm_qty = float(item.get('Revised Norm Qty', 0) or 0)
                stock_value = float(item.get('Current Inventory - VALUE', 0) or 0)
                unit_price = float(item.get('UNIT PRICE', 0) or 0)
                if unit_price == 0 and current_qty > 0:
                    unit_price = stock_value / current_qty
                
                if status_filter == "Excess Inventory" and current_qty > norm_qty:
                    deviation_value = (current_qty - norm_qty) * unit_price
                    vendor_totals[vendor] = vendor_totals.get(vendor, 0.0) + deviation_value
                    vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
                elif status_filter == "Short Inventory" and norm_qty > current_qty:
                    deviation_value = (norm_qty - current_qty) * unit_price
                    vendor_totals[vendor] = vendor_totals.get(vendor, 0.0) + deviation_value
                    vendor_counts[vendor] = vendor_counts.get(vendor, 0) + 1
            except (ValueError, TypeError, ZeroDivisionError):
                continue
        
        if not vendor_totals:
            st.info(f"No vendors found in '{status_filter}' status.")
            return

        combined = [
            (vendor, vendor_totals[vendor], vendor_counts.get(vendor, 0))
            for vendor in vendor_totals
        ]
        
        # Sort and take top N
        top_vendors = sorted(combined, key=lambda x: x[1], reverse=True)[:top_n]
        
        vendor_names = [v[0] for v in top_vendors]
        raw_values = [v[1] for v in top_vendors]
        counts = [v[2] for v in top_vendors]
        
        # Scale values
        values = [v / divisor for v in raw_values]

        # Update labels
        if status_filter == "Excess Inventory":
            y_axis_title = f"Excess Value Above Norm (₹ {unit_type})"
            hover_label = "Excess Value"
        elif status_filter == "Short Inventory":
            y_axis_title = f"Short Value Below Norm (₹ {unit_type})"
            hover_label = "Short Value"
        else:
            y_axis_title = f"Stock Value (₹ {unit_type})"
            hover_label = "Stock Value"

        # Create chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vendor_names,
            y=values,
            marker_color=color,
            text=[f"{hover_label}: ₹{v:,.1f}{suffix}<br>Parts: {c}" for v, c in zip(values, counts)],
            hovertemplate='<b>%{x}</b><br>%{text}<extra></extra>',
        ))
        
        fig.update_layout(
            title=f"{chart_title} (Top {top_n})",
            xaxis_title="Vendor",
            yaxis_title=y_axis_title,
            showlegend=False,
            yaxis=dict(tickformat=".1f", ticksuffix=suffix),
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

# -----------------------------------------------------------------------------
# CLASS: InventoryManagementSystem
# -----------------------------------------------------------------------------
class InventoryManagementSystem:
    """Main application class"""
    
    def __init__(self):
        self.debug = True
        self.analyzer = InventoryAnalyzer()
        self.persistence = DataPersistence()
        self.initialize_session_state()
        self.status_colors = {
            'Within Norms': '#4CAF50',
            'Excess Inventory': '#2196F3',
            'Short Inventory': '#F44336'
        }
        
    def initialize_session_state(self):
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
        self.persistent_keys = [
            'persistent_pfep_data', 'persistent_pfep_locked',
            'persistent_inventory_data', 'persistent_inventory_locked',
            'persistent_analysis_results'
        ]
        for key in self.persistent_keys:
            if key not in st.session_state:
                st.session_state[key] = None

    def safe_float_convert(self, value: Any) -> float:
        if value is None or pd.isna(value) or value == '':
            return 0.0
        try:
            if isinstance(value, str):
                value = re.sub(r'[₹$€£,]', '', value.strip())
                if '%' in value: return float(value.replace('%', '')) / 100
            return float(value)
        except:
            return 0.0

    # ---------------------------------------------------------
    # AUTH & UTILS
    # ---------------------------------------------------------
    def authenticate_user(self):
        st.sidebar.markdown("### 🔐 Authentication")
        if st.session_state.user_role is None:
            role = st.sidebar.selectbox("Select Role", ["Select Role", "Admin", "User"])
            if role == "Admin":
                pwd = st.sidebar.text_input("Password", type="password")
                if st.sidebar.button("Login"):
                    if pwd == "Agilomatrix@123":
                        st.session_state.user_role = "Admin"
                        st.rerun()
                    else:
                        st.error("Invalid password")
            elif role == "User":
                if st.sidebar.button("Enter as User"):
                    st.session_state.user_role = "User"
                    st.rerun()
        else:
            st.sidebar.success(f"Logged in as {st.session_state.user_role}")
            if st.sidebar.button("Logout"):
                st.session_state.clear()
                st.rerun()

    # ---------------------------------------------------------
    # DATA PROCESSING
    # ---------------------------------------------------------
    def standardize_pfep_data(self, df):
        if df is None or df.empty: return []
        
        column_mappings = {
            'part_no': ['part_no', 'part no', 'material', 'item_code'],
            'description': ['description', 'desc', 'part description'],
            'rm_qty': ['rm_in_qty', 'rm_qty', 'required_qty', 'norm_qty'],
            'unit_price': ['unit_price', 'price', 'rate', 'unit cost'],
            'vendor_name': ['vendor_name', 'vendor', 'supplier']
        }
        
        # Basic mapping logic (simplified for brevity)
        df.columns = df.columns.astype(str).str.strip().str.lower()
        mapped = {}
        for key, variations in column_mappings.items():
            for v in variations:
                if v in df.columns:
                    mapped[key] = v
                    break
        
        if 'part_no' not in mapped or 'rm_qty' not in mapped:
            st.error("Missing required columns (Part No, RM Qty)")
            return []

        standardized = []
        for _, row in df.iterrows():
            standardized.append({
                'Part_No': str(row[mapped['part_no']]),
                'Description': str(row.get(mapped.get('description'), '')),
                'RM_IN_QTY': self.safe_float_convert(row[mapped['rm_qty']]),
                'unit_price': self.safe_float_convert(row.get(mapped.get('unit_price'), 1.0)),
                'Vendor_Name': str(row.get(mapped.get('vendor_name'), 'Unknown'))
            })
        return standardized

    def standardize_current_inventory(self, df):
        if df is None or df.empty: return []
        
        column_mappings = {
            'part_no': ['part_no', 'part no', 'material', 'item_code'],
            'current_qty': ['current_qty', 'qty', 'quantity', 'stock'],
            'stock_value': ['stock_value', 'value', 'amount']
        }
        
        df.columns = df.columns.astype(str).str.strip().str.lower()
        mapped = {}
        for key, variations in column_mappings.items():
            for v in variations:
                if v in df.columns:
                    mapped[key] = v
                    break
                    
        if 'part_no' not in mapped or 'current_qty' not in mapped:
            st.error("Missing required columns (Part No, Current Qty)")
            return []

        standardized = []
        for _, row in df.iterrows():
            standardized.append({
                'Part_No': str(row[mapped['part_no']]),
                'Current_QTY': self.safe_float_convert(row[mapped['current_qty']]),
                'Current Inventory - VALUE': self.safe_float_convert(row.get(mapped.get('stock_value'), 0))
            })
        return standardized

    # ---------------------------------------------------------
    # ADMIN & USER UI
    # ---------------------------------------------------------
    def admin_data_management(self):
        st.title("Admin: PFEP Management")
        
        # Tolerance Setting
        if "admin_tolerance" not in st.session_state:
            st.session_state.admin_tolerance = 30
        st.session_state.admin_tolerance = st.selectbox(
            "Tolerance (%)", [0, 10, 20, 30, 40, 50], 
            index=[0, 10, 20, 30, 40, 50].index(st.session_state.admin_tolerance)
        )
        
        uploaded_file = st.file_uploader("Upload PFEP Data", type=['xlsx', 'csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            data = self.standardize_pfep_data(df)
            if data:
                self.persistence.save_data_to_session_state('persistent_pfep_data', data)
                st.success(f"Loaded {len(data)} PFEP records")
                
        if st.button("Lock Data"):
            st.session_state.persistent_pfep_locked = True
            st.success("Data Locked")

    def user_inventory_upload(self):
        st.title("User: Inventory Analysis")
        
        pfep_locked = st.session_state.get('persistent_pfep_locked', False)
        if not pfep_locked:
            st.warning("PFEP data not locked by Admin.")
            return

        # Check if already analyzed
        if st.session_state.get('persistent_analysis_results'):
            self.display_analysis_interface()
            if st.button("Reset Analysis"):
                st.session_state.persistent_analysis_results = None
                st.rerun()
            return

        uploaded_file = st.file_uploader("Upload Inventory", type=['xlsx', 'csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            data = self.standardize_current_inventory(df)
            if data:
                self.persistence.save_data_to_session_state('persistent_inventory_data', data)
                
                # Run Analysis immediately
                pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
                results = self.analyzer.analyze_inventory(pfep_data, data)
                self.persistence.save_data_to_session_state('persistent_analysis_results', results)
                st.rerun()

    def display_analysis_interface(self):
        results = self.persistence.load_data_from_session_state('persistent_analysis_results')
        if not results: return
        
        self.display_enhanced_analysis_charts(results)
        
        # Export
        df = pd.DataFrame(results)
        csv = df.to_csv(index=False)
        st.download_button("Download Report", csv, "inventory_analysis.csv", "text/csv")

    # ---------------------------------------------------------
    # VISUALIZATION (UPDATED)
    # ---------------------------------------------------------
    def display_enhanced_analysis_charts(self, analysis_results):
        """Display enhanced visual summaries with Top N slider and Unit selection"""
        st.subheader("📊 Enhanced Inventory Charts")
        df = pd.DataFrame(analysis_results)
        if df.empty:
            st.warning("⚠️ No data available for charts.")
            return

        # ---------------------------------------------------------
        # 🎛️ CHART SETTINGS CONTROLS
        # ---------------------------------------------------------
        with st.container():
            st.markdown("### ⚙️ Chart Settings")
            col_set1, col_set2 = st.columns(2)
            
            with col_set1:
                # Toggle for Units
                chart_unit = st.radio(
                    "Select Value Unit:",
                    ["Lakhs", "Millions"],
                    horizontal=True,
                    key="chart_unit_selector"
                )
            
            with col_set2:
                # Slider for Top N
                top_n = st.slider(
                    "Number of Items to Display (Top N):",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    key="chart_top_n_slider"
                )

        # 🧮 Determine Divisor and Suffix based on selection
        if chart_unit == "Millions":
            divisor = 1_000_000
            suffix = "M"
        else:  # Lakhs
            divisor = 100_000
            suffix = "L"

        st.markdown("---")

        # ✅ 1. Top Parts by Value
        value_col = None
        for col in ['Current Inventory - VALUE', 'Stock_Value', 'Current Inventory-VALUE']:
            if col in df.columns:
                value_col = col
                break
        
        if value_col and 'PART NO' in df.columns:
            # Filter top N parts with non-zero value
            chart_data = (
                df[df[value_col] > 0]
                .sort_values(by=value_col, ascending=False)
                .head(top_n)  # Uses Slider Value
                .copy()
            )
            
            # Apply Unit Conversion
            chart_data['Value_Scaled'] = chart_data[value_col] / divisor
            
            # Combine description and part no
            desc_col = 'PART DESCRIPTION' if 'PART DESCRIPTION' in df.columns else 'Description'
            chart_data['Part'] = chart_data.apply(
                lambda row: f"{str(row.get(desc_col, ''))[:20]}... ({row['PART NO']})",
                axis=1
            )

            # Determine Status
            if 'Status' in chart_data.columns:
                chart_data['Inventory_Status'] = chart_data['Status']
            elif 'INVENTORY REMARK STATUS' in chart_data.columns:
                chart_data['Inventory_Status'] = chart_data['INVENTORY REMARK STATUS']
            else:
                chart_data['Inventory_Status'] = 'Within Norms' 

            color_map = {
                "Excess Inventory": "#2196F3",
                "Short Inventory": "#F44336", 
                "Within Norms": "#4CAF50"
            }

            # Create chart
            fig1 = go.Figure()
            
            # Add bars
            for i, row in chart_data.iterrows():
                color = color_map.get(row['Inventory_Status'], "#808080")
                fig1.add_trace(go.Bar(
                    x=[row['Part']],
                    y=[row['Value_Scaled']],
                    name=row['Inventory_Status'],
                    marker_color=color,
                    hovertemplate=f"<b>%{{x}}</b><br>Value: ₹%{{y:,.1f}}{suffix}<br>Status: {row['Inventory_Status']}<extra></extra>",
                    showlegend=False
                ))

            # Add Legend Hack
            for status, color in color_map.items():
                fig1.add_trace(go.Bar(x=[None], y=[None], name=status, marker_color=color, showlegend=True))

            fig1.update_layout(
                title=f"Top {top_n} Parts by Stock Value",
                xaxis_title="Parts",
                yaxis_title=f"Stock Value ({chart_unit})",
                xaxis_tickangle=-45,
                yaxis=dict(tickformat=',.1f', ticksuffix=suffix),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig1, use_container_width=True)

        # ✅ 2. Top Vendors by Value
        vendor_col = next((col for col in ['Vendor', 'Vendor Name', 'VENDOR'] if col in df.columns), None)
        if vendor_col and value_col:
            vendor_df = df.groupby(vendor_col)[value_col].sum().reset_index()
            vendor_df = vendor_df.sort_values(by=value_col, ascending=False).head(top_n)
            
            if not vendor_df.empty:
                vendor_df['Value_Scaled'] = vendor_df[value_col] / divisor
                
                fig3 = px.bar(
                    vendor_df,
                    x=vendor_col,
                    y='Value_Scaled',
                    title=f'Top {top_n} Vendors by Stock Value',
                    labels={'Value_Scaled': f'Value ({chart_unit})'},
                    color='Value_Scaled',
                    color_continuous_scale='Viridis'
                )
                fig3.update_layout(
                    yaxis=dict(tickformat=',.1f', ticksuffix=suffix),
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig3, use_container_width=True)

        # ✅ 3. Top Parts by Inventory Status (Excess / Short)
        try:
            st.markdown(f"## 🧩 Top {top_n} Parts by Inventory Status")
            
            status_configs = [
                ("Excess Inventory", "Excess Value", "#2196F3"),
                ("Short Inventory", "Shortage Value", "#F44336")
            ]

            for status, label_prefix, color in status_configs:
                st.subheader(f"{'🔵' if status == 'Excess Inventory' else '🔴'} Top {top_n} {status} Parts")
                
                status_df = df[df['INVENTORY REMARK STATUS'] == status].copy()
                
                if status_df.empty:
                    st.info(f"No {status} found.")
                    continue

                if status == "Excess Inventory":
                    # Filter positive deviation
                    status_df = status_df[status_df['Stock Deviation Value'] > 0]
                    col_to_plot = 'Stock Deviation Value'
                else:
                    # Filter negative deviation and take absolute
                    status_df = status_df[status_df['Stock Deviation Value'] < 0]
                    status_df['Abs_Deviation'] = abs(status_df['Stock Deviation Value'])
                    col_to_plot = 'Abs_Deviation'

                # Sort and slice using Slider Value
                status_df = status_df.sort_values(by=col_to_plot, ascending=False).head(top_n)
                
                # Apply Unit Scaling
                status_df['Value_Scaled'] = status_df[col_to_plot] / divisor
                desc_col = 'PART DESCRIPTION' if 'PART DESCRIPTION' in df.columns else 'Description'
                status_df['Part_Label'] = status_df[desc_col].astype(str).str[:20] + "... (" + status_df['PART NO'].astype(str) + ")"

                fig = px.bar(
                    status_df,
                    x='Part_Label',
                    y='Value_Scaled',
                    color_discrete_sequence=[color],
                    title=f"Top {top_n} {status} ({chart_unit})"
                )
                fig.update_layout(
                    xaxis_title="Part Description",
                    yaxis_title=f"{label_prefix} ({chart_unit})",
                    yaxis=dict(tickformat=',.1f', ticksuffix=suffix),
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True, key=f"{status}_{chart_unit}_chart")

        except Exception as e:
            st.error(f"Error displaying status charts: {str(e)}")

        # ✅ 4. Top Vendors by Status
        try:
            st.markdown(f"## 🏢 Top {top_n} Vendors by Inventory Status")
            
            self.analyzer.show_vendor_chart_by_status(
                processed_data=analysis_results,
                status_filter="Excess Inventory",
                chart_title=f"Top {top_n} Vendors - Excess Value",
                chart_key="excess_vendors_dynamic",
                color=self.status_colors["Excess Inventory"],
                unit_type=chart_unit,
                top_n=top_n
            )

            self.analyzer.show_vendor_chart_by_status(
                processed_data=analysis_results,
                status_filter="Short Inventory",
                chart_title=f"Top {top_n} Vendors - Short Value",
                chart_key="short_vendors_dynamic",
                color=self.status_colors["Short Inventory"],
                unit_type=chart_unit,
                top_n=top_n
            )

        except Exception as e:
            st.error(f"Error displaying vendor status charts: {str(e)}")

    def run(self):
        st.sidebar.image("https://img.icons8.com/color/96/000000/warehouse.png", width=60)
        self.authenticate_user()
        
        if st.session_state.user_role == "Admin":
            self.admin_data_management()
        elif st.session_state.user_role == "User":
            self.user_inventory_upload()
        else:
            st.info("Please login to continue.")

if __name__ == "__main__":
    app = InventoryManagementSystem()
    app.run()
