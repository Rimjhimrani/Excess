import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import pickle
import base64
import uuid
import io

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


# Custom CSS for better styling
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

.status-card {
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
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

.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
}

.lock-button {
    background-color: #28a745;
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    border: none;
    font-weight: bold;
}

.switch-user-button {
    background-color: #007bff;
    color: white;
    padding: 8px 16px;
    border-radius: 5px;
    border: none;
    font-weight: bold;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

class DataPersistence:
    """Handle data persistence across sessions"""
    
    @staticmethod
    def save_data_to_session_state(key, data):
        """Save data with timestamp to session state"""
        st.session_state[key] = {
            'data': data,
            'timestamp': datetime.now(),
            'saved': True
        }
    
    @staticmethod
    def load_data_from_session_state(key):
        """Load data from session state if it exists"""
        if key in st.session_state and isinstance(st.session_state[key], dict):
            return st.session_state[key].get('data')
        return None
    
    @staticmethod
    def is_data_saved(key):
        """Check if data is saved"""
        if key in st.session_state and isinstance(st.session_state[key], dict):
            return st.session_state[key].get('saved', False)
        return False
    
    @staticmethod
    def get_data_timestamp(key):
        """Get data timestamp"""
        if key in st.session_state and isinstance(st.session_state[key], dict):
            return st.session_state[key].get('timestamp')
        return None

class InventoryAnalyzer:
    """Enhanced inventory analysis with comprehensive reporting"""
    
    def __init__(self):
        self.persistence = self
        self.status_colors = {
            'Within Norms': '#4CAF50',    # Green
            'Excess Inventory': '#2196F3', # Blue
            'Short Inventory': '#F44336'   # Red
        }
        
    def analyze_inventory(self, pfep_data, current_inventory, tolerance=None):
        """Analyze ONLY inventory parts that exist in PFEP and apply cleaned output format."""
        if tolerance is None:
            tolerance = st.session_state.get("admin_tolerance", 30)  # fallback
        results = []
        # Normalize and create lookup dictionaries
        pfep_dict = {str(item['Part_No']).strip().upper(): item for item in pfep_data}
        inventory_dict = {str(item['Part_No']).strip().lower(): item for item in current_inventory}  # BUG: case mismatch
        # ✅ Loop only through inventory items that exist in PFEP
        for part_no, inventory_item in inventory_dict.items():
            pfep_item = pfep_dict.get(part_no)
            if not pfep_item:
                continue
            # Extract values safely
            current_qty = float(inventory_item.get('Current_QTY', 0))
            stock_value = float(inventory_item.get('Stock_Value', 0))
            rm_qty = float(pfep_item.get('RM_IN_QTY', 0))
            unit_price = float(pfep_item.get('Unit_Price', 0))
            rm_days = pfep_item.get('RM_IN_DAYS', '')

            # Short/Excess Inventory calculation
            short_excess_qty = current_qty - rm_qty
            value = short_excess_qty / unit_price  # BUG: division instead of multiplication

            # Status (with tolerance logic applied to % difference)
            if rm_qty > 0:
                variance_pct = ((current_qty - rm_qty) / rm_qty) * 100
            else:
                variance_pct = 0
            if abs(variance_pct) <= tolerance:
                status = 'Within Norms'
            elif variance_pct > tolerance:
                status = 'Excess Inventory'
            else:
                status = 'Short Inventory'
            # ✅ Build final cleaned result
            result = {
                'PART NO': part_no,
                'PART DESCRIPTION': pfep_item.get('Description', ''),
                'Current Inventory-QTY': current_qty,
                'Inventory Norms - QTY': rm_qty,
                'Current Inventory - VALUE': stock_value,
                'SHORT/EXCESS INVENTORY': short_excess_qty,
                'INVENTORY REMARK STATUS': status,
                'Status': status,  # ✅ Add this line
                'VALUE(Unit Price* Short/Excess Inventory)': value,
                'UNIT PRICE': unit_price,
                'RM IN DAYS': rm_days,
                'Vendor Name': pfep_item.get('Vendor_Name', 'Unknown'),
                'Vendor_Code': pfep_item.get('Vendor_Code', ''),
                'City': pfep_item.get('City', ''),
                'State': pfep_item.get('State', '')
            }
            results.append(result)
        return results
        
    def run(self):
        st.title("Inventory Analyzer")
        st.write("This is a test run for InventoryAnalyzer.")

    def get_vendor_summary(self, processed_data):
        """Summarize inventory by vendor using actual Stock_Value field from the file."""
        from collections import defaultdict
        summary = defaultdict(lambda: {
            'total_parts': 0,
            'short_parts': 0,
            'excess_parts': 0,
            'normal_parts': 0,
            'total_value': 0.0
        })
        for item in processed_data:
            vendor = item.get('Vendor Name', 'Unknown')
            status = item.get('INVENTORY REMARK STATUS', 'Unknown')
            stock_value = item.get('Stock_Value') or item.get('Current Inventory - VALUE') or 0
            try:
                stock_value = float(stock_value)
            except (ValueError, TypeError):
                stock_value = 0.0
            summary[vendor]['total_parts'] += 1
            summary[vendor]['total_value'] += stock_value
            if status == "Short Inventory":  # Fixed status name
                summary[vendor]['short_parts'] += 1
            elif status == "Excess Inventory":  # Fixed status name
                summary[vendor]['excess_parts'] += 1
            elif status == "Within Norms":
                summary[vendor]['normal_parts'] += 1
        # Fixed: Return statement moved outside the loop
        return summary
    def show_vendor_chart_by_status(self, processed_data, status_filter, chart_title, chart_key, color):
        """Show top 10 vendors filtered by inventory remark status (short, excess, within norms)"""
        from collections import defaultdict
        # Filter by inventory status
        filtered = [item for item in processed_data if item.get('INVENTORY REMARK STATUS') == status_filter]
        # Sum Stock Value by Vendor
        vendor_totals = defaultdict(float)
        for item in filtered:
            vendor = item.get('Vendor Name', 'Unknown')
            try:
                # Fixed: Use 'item' instead of undefined 'row' variable
                stock_value = item.get('Stock_Value', 0) or item.get('Current Inventory - VALUE', 0)
                stock_value = float(stock_value) if stock_value else 0.0
            except (ValueError, TypeError):
                stock_value = 0.0
            vendor_totals[vendor] += stock_value
        # Sort top 10
        sorted_vendors = sorted(vendor_totals.items(), key=lambda x: x[1], reverse=True)[:5]  # BUG: only top 5 instead of 10
        if not sorted_vendors:
            st.info(f"No vendors found in '{status_filter}'")
            return
        vendor_names = [v[0] for v in sorted_vendors]
        stock_values = [v[1] for v in sorted_vendors]
        # Plot chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=vendor_names, y=stock_values, marker_color=color))
        fig.update_layout(
            title=chart_title,
            xaxis_title="Vendor",
            yaxis_title="Stock Value (₹)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

class InventoryManagementSystem:
    """Main application class"""
    
    def __init__(self):
        self.debug = True
        self.analyzer = InventoryAnalyzer()
        self.persistence = DataPersistence()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
        
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'default_tolerance': 30,
                'chart_theme': 'plotly'
            }
        
        # Initialize persistent data keys
        self.persistent_keys = [
            'persistent_pfep_data',
            'persistent_pfep_locked',
            'persistent_inventory_data', 
            'persistent_inventory_locked',
            'persistent_analysis_results'
        ]
        
        # Initialize persistent data if not exists
        for key in self.persistent_keys:
            if key not in st.session_state:
                st.session_state[key] = []  # BUG: should be None, not empty list
    def safe_print(self, message):
        """Safely print to streamlit or console"""
        try:
            st.write(message)
        except NameError:
            print(message)
    
    def safe_error(self, message):
        """Safely show error in streamlit or console"""
        try:
            st.error(message)
        except NameError:
            print(f"ERROR: {message}")
    
    def safe_warning(self, message):
        """Safely show warning in streamlit or console"""
        try:
            st.warning(message)
        except NameError:
            print(f"WARNING: {message}")
    
    def safe_float_convert(self, value):
        """Enhanced float conversion with optional debug output via Streamlit."""
        if pd.isna(value) or value in ['', None]:
            if self.debug:
                st.write("🟡 safe_float_convert: NULL or empty value → 0.0")
            return 0.0
        try:
            if isinstance(value, (int, float)):
                result = float(value)
                if self.debug:
                    st.write(f"✅ Converted numeric: {value} → {result}")
                return result
            str_value = str(value).strip()
            str_value = str_value.replace(',', '').replace(' ', '').replace('₹', '').replace('$', '').replace('€', '')
            if str_value.endswith('%'):
                str_value = str_value[:-1]
            if str_value.startswith('(') and str_value.endswith(')'):
                str_value = '-' + str_value[1:-1]
            result = float(str_value)
            if self.debug:
                st.write(f"✅ Parsed '{value}' → {result}")
            return result
        except (ValueError, TypeError) as e:
            if self.debug:
                st.write(f"❌ Error converting '{value}' → 0.0 | Error: {e}")
            return 1.0  # BUG: returns 1.0 instead of 0.0

            
    def safe_int_convert(self, value):
        """Enhanced safe int conversion"""
        return int(self.safe_float_convert(value))
            
    def create_top_parts_chart(self, data, status_type, color, key):
        """Display top 10 parts by absolute value of short/excess inventory impact (₹)."""
        # Filter top parts by selected inventory status
        top_items = [
            item for item in data 
            if item['INVENTORY REMARK STATUS'] == status_type
        ]
        # Sort by absolute ₹ value of short/excess impact
        top_items = sorted(
            top_items,
            key=lambda x: abs(x['VALUE(Unit Price* Short/Excess Inventory)']),
            reverse=False  # BUG: should be True for descending order
        )[:10]
        
        if not top_items:
            st.info(f"No parts found for status: {status_type}")
            return
        part_nos = [item['PART NO'] for item in top_items]
        values = [item['VALUE(Unit Price* Short/Excess Inventory)'] for item in top_items]
        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(x=values, y=part_nos, orientation='h', marker_color=color)
        ])
        fig.update_layout(
            title=f"Top 10 Parts by Value - {status_type}",
            xaxis_title="Inventory Value Impact (₹)",
            yaxis_title="Part Number",
            yaxis=dict(autorange='reversed')
        )
        
        st.plotly_chart(fig, use_container_width=True, key=key)
 
    def authenticate_user(self):
        """Enhanced authentication system with better UX and user switching"""
        st.sidebar.markdown("### 🔐 Authentication")
        
        if st.session_state.user_role is None:
            role = st.sidebar.selectbox(
                "Select Role", 
                ["Select Role", "Admin", "User"],
                help="Choose your role to access appropriate features"
            )
            
            if role == "Admin":
                with st.sidebar.container():
                    st.markdown("**Admin Login**")
                    password = st.text_input("Admin Password", type="password", key="admin_pass")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔑 Login", key="admin_login"):
                            if password == "Agilomatrix@123":  # BUG: wrong password
                                st.session_state.user_role = "Admin"
                                st.success("✅ Admin authenticated!")
                                st.rerun()
                            else:
                                st.error("❌ Invalid password")
                    with col2:
                        if st.button("🏠 Demo", key="admin_demo"):
                            st.session_state.user_role = "Admin"
                            st.info("🎮 Demo mode activated!")
                            st.rerun()
            
            elif role == "User":
                if st.sidebar.button("👤 Enter as User", key="user_login"):
                    st.session_state.user_role = "User"
                    st.sidebar.success("✅ User access granted!")
                    st.rerun()
        else:
            # User info and controls
            st.sidebar.success(f"✅ **{st.session_state.user_role}** logged in")
            
            # Display data status
            self.display_data_status()
            
            # User switching option for Admin
            if st.session_state.user_role == "Admin":
                # ✅ Show PFEP lock status
                pfep_locked = st.session_state.get("persistent_pfep_locked", False)
                st.sidebar.markdown(f"🔒 PFEP Locked: **{pfep_locked}**")
                # ✅ Always show switch role if PFEP is locked
                if pfep_locked:
                    st.sidebar.markdown("### 🔄 Switch Role")
                    if st.sidebar.button("👤 Switch to User View", key="switch_to_user"):
                        st.session_state.user_role = "User"
                        st.sidebar.success("✅ Switched to User view!")
                        st.rerun()
                else:
                    st.sidebar.info("ℹ️ PFEP is not locked. Lock PFEP to allow switching to User.")

            
            # User preferences (for Admin only)
            if st.session_state.user_role == "Admin":
                with st.sidebar.expander("⚙️ Preferences"):
                    st.session_state.user_preferences['default_tolerance'] = st.selectbox(
                        "Default Tolerance", [10, 20, 30, 40, 50], 
                        index=2, key="pref_tolerance"
                    )
                    st.session_state.user_preferences['chart_theme'] = st.selectbox(
                        "Chart Theme", ['plotly', 'plotly_white', 'plotly_dark'],
                        key="pref_theme"
                    )
            
            # Logout button
            st.sidebar.markdown("---")
            if st.sidebar.button("🚪 Logout", key="logout_btn"):
                # Only clear user session, not persistent data
                keys_to_keep = self.persistent_keys + ['user_preferences']
                session_copy = {k: v for k, v in st.session_state.items() if k in keys_to_keep}
                
                # Clear all session state
                st.session_state.clear()
                
                # Restore persistent data
                for k, v in session_copy.items():
                    st.session_state[k] = v
                
                st.rerun()
    
    def display_data_status(self):
        """Display current data loading status in sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Data Status")
        
        # Check persistent PFEP data
        pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
        pfep_locked = st.session_state.get('persistent_pfep_locked', False)
        
        if pfep_data:
            pfep_count = len(pfep_data)
            lock_icon = "🔒" if pfep_locked else "🔓"
            st.sidebar.success(f"✅ PFEP Data: {pfep_count} parts {lock_icon}")
            timestamp = self.persistence.get_data_timestamp('persistent_pfep_data')
            if timestamp:
                st.sidebar.caption(f"Loaded: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.sidebar.error("❌ PFEP Data: Not loaded")
        
        # Check persistent inventory data
        inventory_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
        inventory_locked = st.session_state.get('persistent_inventory_locked', False)
        
        if inventory_data:
            inv_count = len(inventory_data)
            lock_icon = "🔒" if inventory_locked else "🔓"
            st.sidebar.success(f"✅ Inventory: {inv_count} parts {lock_icon}")
            timestamp = self.persistence.get_data_timestamp('persistent_inventory_data')
            if timestamp:
                st.sidebar.caption(f"Loaded: {timestamp.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.sidebar.error("❌ Inventory: Not loaded")
        
        # Analysis results status
        analysis_data = self.persistence.load_data_from_session_state('persistent_analysis_results')
        if analysis_data:
            st.sidebar.info(f"📈 Analysis: {len(analysis_data)} parts analyzed")
    
    def load_sample_pfep_data(self):
        pfep_sample = [
            ["AC0303020106", "FLAT ALUMINIUM PROFILE", 4.000, "V001", "Vendor_A", "Mumbai", "Maharashtra"],
            # ... (your full list unchanged)
            ["JJ1010101010", "WINDSHIELD WASHER", 25, "V002", "Vendor_B", "Delhi", "Delhi"]
        ]
        pfep_data = []
        for row in pfep_sample:
            pfep_data.append({
                'Part_No': row[0],
                'Description': row[1],
                'RM_IN_QTY': self.safe_float_convert(row[2]),
                'Vendor_Code': row[3],
                'Vendor_Name': row[4],
                'City': row[5],
                'State': row[6],
                'Unit_Price': 100,            # 🔁 you can customize this per part
                'RM_IN_DAYS': 7               # 🔁 default or configurable
            })
        return pfep_data
    
    def load_sample_current_inventory(self):
        """Load sample current inventory data with consistent fields"""
        current_sample = [
            ["AC0303020106", "FLAT ALUMINIUM PROFILE", 5.230, 496],
            # ... rest of your data
            ["JJ1010101010", "WINDSHIELD WASHER", 33, 495]
        ]
        return [{
            'Part_No': row[0],
            'Description': row[1],
            'Current_QTY': self.safe_float_convert(row[2]),
            'Stock_Value': self.safe_float_convert(row[3])
        } for row in current_sample]
    
    def standardize_pfep_data(self, df):
        """Enhanced PFEP data standardization with added Unit_Price and RM_IN_DAYS support"""
        if df is None or df.empty:
            return []
        # Column mapping with extended support
        column_mappings = {
            'part_no': ['part_no', 'part_number', 'material', 'material_code', 'item_code', 'code', 'part no', 'partno'],
            'description': ['description', 'item_description', 'part_description', 'desc', 'part description', 'material_description', 'item desc'],
            'rm_qty': ['rm_in_qty', 'rm_qty', 'required_qty', 'norm_qty', 'target_qty', 'rm', 'ri_in_qty', 'rm in qty'],
            'rm_days': ['rm_in_days', 'rm days', 'inventory days', 'rmindays'],
            'unit_price': ['unit_price', 'price', 'unit cost', 'unit rate', 'unitprice'],
            'vendor_code': ['vendor_code', 'vendor_id', 'supplier_code', 'supplier_id', 'vendor id', 'Vendor Code', 'vendor code'],
            'vendor_name': ['vendor_name', 'vendor', 'supplier_name', 'supplier', 'Vendor Name', 'vendor name'],
            'city': ['city', 'location', 'place'],
            'state': ['state', 'region', 'province']
        }
        # Normalize and map columns
        df_columns = [col.lower().strip() for col in df.columns]
        mapped_columns = {}
        for key, variations in column_mappings.items():
            for variation in variations:
                if variation in df_columns:
                    original_col = df.columns[df_columns.index(variation)]
                    mapped_columns[key] = original_col
                    break
        # Check for required columns
        if 'part_no' not in mapped_columns or 'rm_qty' not in mapped_columns:
            st.error("❌ Required columns not found. Please ensure your file has Part Number and RM Quantity columns.")
            return []
        standardized_data = []
        for _, row in df.iterrows():
            item = {
                'Part_No': str(row[mapped_columns['part_no']]).strip(),
                'Description': str(row.get(mapped_columns.get('description', ''), '')).strip(),
                'RM_IN_QTY': self.safe_float_convert(row[mapped_columns['rm_qty']]),
                'RM_IN_DAYS': self.safe_float_convert(row.get(mapped_columns.get('rm_days', ''), 0)),
                'Unit_Price': self.safe_float_convert(row.get(mapped_columns.get('unit_price', ''), 0)),
                'Vendor_Code': str(row.get(mapped_columns.get('vendor_code', ''), '')).strip(),
                'Vendor_Name': str(row.get(mapped_columns.get('vendor_name', ''), 'Unknown')).strip(),
                'City': str(row.get(mapped_columns.get('city', ''), '')).strip(),
                'State': str(row.get(mapped_columns.get('state', ''), '')).strip()
            }
            standardized_data.append(item)
        return standardized_data
    
    def standardize_current_inventory(self, df):
        """Standardize current inventory data with full column mappings and debugging."""
        if df is None or df.empty:
            return []
        # 🔁 Add all possible column mappings
        column_mappings = {
            'part_no': ['part_no', 'part_number', 'material', 'material_code', 'item_code', 'code'],
            'description': ['description', 'item_description', 'part_description', 'desc'],
            'current_qty': ['current_qty', 'qty', 'quantity', 'stock_qty', 'available_qty'],
            'stock_value': ['stock_value', 'value', 'total_value', 'inventory_value', 'stock value', 'Stock Value'],
            'uom': ['uom', 'unit', 'unit_of_measure'],
            'location': ['location', 'store', 'warehouse', 'site'],
            'vendor_code': ['vendor_code', 'vendor_id', 'supplier_code', 'supplier_id', 'vendor id', 'Vendor Code', 'vendor code'],
            'batch': ['batch', 'batch_number', 'lot', 'lot_number']
        }
        df_columns_lower = {col.lower().strip(): col for col in df.columns if col is not None}
        mapped_columns = {}
        for key, variations in column_mappings.items():
            for variation in variations:
                if variation.lower() in df_columns_lower:
                    mapped_columns[key] = df_columns_lower[variation.lower()]
                    break
        # Debug: show mappings
        if self.debug:
            st.write("🔍 DEBUG: Column mappings found:")
            for key, col in mapped_columns.items():
                st.write(f"  {key} → {col}")
        if 'part_no' not in mapped_columns or 'current_qty' not in mapped_columns:
            st.error("❌ Required columns not found. Please ensure your file has Part Number and Current Quantity columns.")
            return []
        standardized_data = []
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                part_no = str(row[mapped_columns['part_no']]).strip()
                if part_no.lower() in ('nan', '', 'none'):
                    continue
                item = {
                    'Part_No': part_no,
                    'Current_QTY': self.safe_float_convert(row[mapped_columns['current_qty']]),
                    'Stock_Value': self.safe_float_convert(row.get(mapped_columns.get('stock_value', ''), 0)),
                    'Description': str(row.get(mapped_columns.get('description', ''), '')).strip(),
                    'UOM': str(row.get(mapped_columns.get('uom', ''), '')).strip(),
                    'Location': str(row.get(mapped_columns.get('location', ''), '')).strip(),
                    'Vendor_Code': str(row.get(mapped_columns.get('vendor_code', ''), '')).strip(),
                    'Batch': str(row.get(mapped_columns.get('batch', ''), '')).strip()
                }
                standardized_data.append(item)
                if self.debug and i < 5:
                    st.write(f"🔍 Row {i+1}: {item}")
            except Exception as e:
                if self.debug:
                    st.write(f"⚠️ Error processing row {i+1}: {e}")
                continue
        if self.debug:
            st.write(f"✅ Total standardized records: {len(standardized_data)}")
        return standardized_data
    
    def validate_inventory_against_pfep(self, inventory_data):
        """Validate inventory data against PFEP master data with normalized keys and warnings."""
        pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
        if not pfep_data:
            return {'is_valid': False, 'issues': ['No PFEP data available'], 'warnings': []}
        # Normalize part numbers
        def normalize(pn): return str(pn).strip().upper()
        pfep_df = pd.DataFrame(pfep_data)
        inventory_df = pd.DataFrame(inventory_data)
        pfep_df['Part_No'] = pfep_df['Part_No'].apply(normalize)
        inventory_df['Part_No'] = inventory_df['Part_No'].apply(normalize)

        pfep_parts = set(pfep_df['Part_No'])
        inventory_parts = set(inventory_df['Part_No'])

        issues = []
        warnings = []

        missing_parts = pfep_parts - inventory_parts
        extra_parts = inventory_parts - pfep_parts

        if missing_parts:
            warnings.append(f"Parts missing in inventory: {len(missing_parts)} parts")
        if extra_parts:
            warnings.append(f"Extra parts in inventory not in PFEP: {len(extra_parts)} parts")
        # Check for parts with zero quantity
        zero_qty_parts = inventory_df[inventory_df['Current_QTY'] == 0]['Part_No'].tolist()
        if zero_qty_parts:
            warnings.append(f"Parts with zero quantity: {len(zero_qty_parts)} parts")
        is_valid = len(issues) == 0
        return {
            'is_valid': is_valid,
            'issues': issues,
            'warnings': warnings,
            'pfep_parts_count': len(pfep_parts),
            'inventory_parts_count': len(inventory_parts),
            'matching_parts_count': len(pfep_parts & inventory_parts),
            'missing_parts_count': len(missing_parts),
            'extra_parts_count': len(extra_parts),
            'missing_parts_list': list(missing_parts),
            'extra_parts_list': list(extra_parts),
            'zero_qty_parts_list': zero_qty_parts
        }
    def admin_data_management(self):
        """Admin-only PFEP data management interface"""
        st.header("🔧 Admin Dashboard - PFEP Data Management")
        
        # Check if PFEP data is locked
        pfep_locked = st.session_state.get('persistent_pfep_locked', False)
        
        if pfep_locked:
            st.warning("🔒 PFEP data is currently locked. Users are working with this data.")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.info("To modify PFEP data, first unlock it. This will reset all user analysis.")
            with col2:
                if st.button("🔓 Unlock Data", type="secondary"):
                    st.session_state.persistent_pfep_locked = False
                    # Clear related data when PFEP is unlocked
                    st.session_state.persistent_inventory_data = None
                    st.session_state.persistent_inventory_locked = False
                    st.session_state.persistent_analysis_results = None
                    st.success("✅ PFEP data unlocked. Users need to re-upload inventory data.")
                    st.rerun()
            with col3:
                if st.button("👤 Go to User View", type="primary", help="Switch to user interface"):
                    st.session_state.user_role = "User"
                    st.rerun()
            
            # Display current PFEP data if available
            pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
            if pfep_data:
                self.display_pfep_data_preview(pfep_data)
            return
        # Tolerance Setting for Admin
        st.subheader("📐 Set Analysis Tolerance (Admin Only)")
        # Initialize admin_tolerance if not exists
        if "admin_tolerance" not in st.session_state:
            st.session_state.admin_tolerance = 30
    
        # Create selectbox with proper callback
        new_tolerance = st.selectbox(
            "Tolerance Zone (+/-)",
            options=[10, 20, 30, 40, 50],
            index=[10, 20, 30, 40, 50].index(st.session_state.admin_tolerance),
            format_func=lambda x: f"{x}%",
            key="tolerance_selector"
        )
        # Update tolerance if changed
        if new_tolerance != st.session_state.admin_tolerance:
            st.session_state.admin_tolerance = new_tolerance
            st.success(f"✅ Tolerance updated to {new_tolerance}%")
            
            # If analysis exists, refresh it with new tolerance
            if st.session_state.get('persistent_analysis_results'):
                st.info("🔄 Analysis will be refreshed with new tolerance on next run")
        
        st.markdown(f"**Current Tolerance:** {st.session_state.admin_tolerance}%")
        st.markdown("---")
        
        # PFEP Data Management Section
        st.subheader("📊 PFEP Master Data Management")
        
        # Tab interface for different data input methods
        tab1, tab2, tab3 = st.tabs(["📁 Upload File", "🧪 Load Sample", "📋 Current Data"])
        
        with tab1:
            st.markdown("**Upload PFEP Excel/CSV File**")
            uploaded_file = st.file_uploader(
                "Choose PFEP file",
                type=['xlsx', 'xls', 'csv'],
                help="Upload your PFEP master data file"
            )
            
            if uploaded_file is not None:
                try:
                    # Read file based on extension
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ File loaded: {len(df)} rows")
                    
                    # Show preview
                    with st.expander("📋 Preview Raw Data"):
                        st.dataframe(df.head())
                    
                    # Standardize data
                    standardized_data = self.standardize_pfep_data(df)
                    
                    if standardized_data:
                        st.success(f"✅ Standardized: {len(standardized_data)} valid records")
                        
                        # Show standardized preview
                        with st.expander("📋 Preview Standardized Data"):
                            preview_df = pd.DataFrame(standardized_data[:5])
                            st.dataframe(preview_df)
                        
                        # Save button
                        if st.button("💾 Save PFEP Data", type="primary"):
                            self.persistence.save_data_to_session_state(
                                'persistent_pfep_data', 
                                standardized_data
                            )
                            st.success("✅ PFEP data saved successfully!")
                            st.rerun()
                    else:
                        st.error("❌ No valid data found after standardization")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
        
        with tab2:
            st.markdown("**Load Sample PFEP Data for Testing**")
            st.info("This will load pre-configured sample data for demonstration")
            
            if st.button("🧪 Load Sample PFEP Data", type="secondary"):
                sample_data = self.load_sample_pfep_data()
                self.persistence.save_data_to_session_state(
                    'persistent_pfep_data', 
                    sample_data
                )
                st.success(f"✅ Sample PFEP data loaded: {len(sample_data)} parts")
                st.rerun()
        
        with tab3:
            st.markdown("**Current PFEP Data Status**")
            pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
            
            if pfep_data:
                self.display_pfep_data_preview(pfep_data)
                
                # Lock data for users
                st.markdown("---")
                st.markdown("**🔒 Lock Data for Users**")
                st.info("Locking PFEP data allows users to upload inventory and perform analysis")
                
                if st.button("🔒 Lock PFEP Data", type="primary"):
                    st.session_state.persistent_pfep_locked = True
                    st.success("✅ PFEP data locked! Users can now upload inventory data.")
                    st.rerun()
            else:
                st.warning("❌ No PFEP data available. Please upload or load sample data first.")
    
    def display_pfep_data_preview(self, pfep_data):
        """Display PFEP data preview with statistics"""
        st.markdown("**📊 PFEP Data Overview**")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Parts", len(pfep_data))
        with col2:
            vendors = set(item.get('Vendor_Name', 'Unknown') for item in pfep_data)
            st.metric("Unique Vendors", len(vendors))
        with col3:
            total_rm_qty = sum(item.get('RM_IN_QTY', 0) for item in pfep_data)
            st.metric("Total RM Qty", f"{total_rm_qty:,.0f}")
        with col4:
            avg_unit_price = sum(item.get('Unit_Price', 0) for item in pfep_data) / len(pfep_data)
            st.metric("Avg Unit Price", f"₹{avg_unit_price:.2f}")
        
        # Data preview
        with st.expander("📋 Data Preview (First 10 rows)"):
            preview_df = pd.DataFrame(pfep_data[:10])
            st.dataframe(preview_df)
        
        # Vendor summary
        with st.expander("📈 Vendor Summary"):
            vendor_summary = {}
            for item in pfep_data:
                vendor = item.get('Vendor_Name', 'Unknown')
                if vendor not in vendor_summary:
                    vendor_summary[vendor] = {'count': 0, 'total_qty': 0}
                vendor_summary[vendor]['count'] += 1
                vendor_summary[vendor]['total_qty'] += item.get('RM_IN_QTY', 0)
            
            vendor_df = pd.DataFrame([
                {'Vendor': k, 'Parts Count': v['count'], 'Total RM Qty': v['total_qty']}
                for k, v in vendor_summary.items()
            ])
            st.dataframe(vendor_df)
    
    def user_inventory_upload(self):
        """User interface for inventory upload and analysis"""
        st.header("📦 Inventory Analysis System")
        
        # Check if PFEP data is available and locked
        pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
        pfep_locked = st.session_state.get('persistent_pfep_locked', False)
        
        if not pfep_data or not pfep_locked:
            st.error("❌ PFEP master data is not available or not locked by admin.")
            st.info("Please contact admin to load and lock PFEP data first.")
            return
        
        # Display PFEP status
        st.success(f"✅ PFEP Master Data: {len(pfep_data)} parts available")
        
        # Check if inventory is already loaded and locked
        inventory_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
        inventory_locked = st.session_state.get('persistent_inventory_locked', False)
        
        if inventory_locked and inventory_data:
            st.info("🔒 Inventory data is locked. Proceeding to analysis...")
            self.display_analysis_interface()
            return
        
        # Inventory upload interface
        st.subheader("📊 Upload Current Inventory Data")
        
        # Tab interface
        tab1, tab2 = st.tabs(["📁 Upload File", "🧪 Load Sample"])
        
        with tab1:
            st.markdown("**Upload Current Inventory Excel/CSV File**")
            uploaded_file = st.file_uploader(
                "Choose inventory file",
                type=['xlsx', 'xls', 'csv'],
                help="Upload your current inventory data file"
            )
            
            if uploaded_file is not None:
                try:
                    # Read file
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ File loaded: {len(df)} rows")
                    
                    # Show preview
                    with st.expander("📋 Preview Raw Data"):
                        st.dataframe(df.head())
                    
                    # Standardize data
                    standardized_data = self.standardize_current_inventory(df)
                    
                    if standardized_data:
                        st.success(f"✅ Standardized: {len(standardized_data)} valid records")
                        
                        # Validate against PFEP
                        validation_result = self.validate_inventory_against_pfep(standardized_data)
                        
                        # Display validation results
                        self.display_validation_results(validation_result)
                        
                        if validation_result['is_valid']:
                            # Show standardized preview
                            with st.expander("📋 Preview Standardized Data"):
                                preview_df = pd.DataFrame(standardized_data[:5])
                                st.dataframe(preview_df)
                            
                            # Save and lock button
                            if st.button("💾 Save & Lock Inventory Data", type="primary"):
                                self.persistence.save_data_to_session_state(
                                    'persistent_inventory_data', 
                                    standardized_data
                                )
                                st.session_state.persistent_inventory_locked = True
                                st.success("✅ Inventory data saved and locked!")
                                st.rerun()
                        else:
                            st.error("❌ Please fix validation issues before proceeding")
                    else:
                        st.error("❌ No valid data found after standardization")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
        
        with tab2:
            st.markdown("**Load Sample Inventory Data for Testing**")
            st.info("This will load pre-configured sample inventory data")
            
            if st.button("🧪 Load Sample Inventory Data", type="secondary"):
                sample_data = self.load_sample_current_inventory()
                
                # Validate sample data
                validation_result = self.validate_inventory_against_pfep(sample_data)
                self.display_validation_results(validation_result)
                
                if validation_result['is_valid']:
                    self.persistence.save_data_to_session_state(
                        'persistent_inventory_data', 
                        sample_data
                    )
                    st.session_state.persistent_inventory_locked = True
                    st.success(f"✅ Sample inventory data loaded and locked: {len(sample_data)} parts")
                    st.rerun()
    
    def display_validation_results(self, validation_result):
        """Display inventory validation results"""
        st.markdown("**📋 Validation Results**")
        
        if validation_result['is_valid']:
            st.success("✅ Inventory data validation passed!")
        else:
            st.error("❌ Inventory data validation failed!")
            for issue in validation_result['issues']:
                st.error(f"• {issue}")
        
        # Display warnings
        if validation_result['warnings']:
            st.warning("⚠️ Validation Warnings:")
            for warning in validation_result['warnings']:
                st.warning(f"• {warning}")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("PFEP Parts", validation_result['pfep_parts_count'])
        with col2:
            st.metric("Inventory Parts", validation_result['inventory_parts_count'])
        with col3:
            st.metric("Matching Parts", validation_result['matching_parts_count'])
        
        # Detailed breakdown
        with st.expander("📊 Detailed Breakdown"):
            col1, col2 = st.columns(2)
            with col1:
                if validation_result['missing_parts_count'] > 0:
                    st.warning(f"Missing Parts ({validation_result['missing_parts_count']}):")
                    st.text("\n".join(validation_result['missing_parts_list'][:10]))
                    if len(validation_result['missing_parts_list']) > 10:
                        st.text(f"... and {len(validation_result['missing_parts_list']) - 10} more")
            
            with col2:
                if validation_result['extra_parts_count'] > 0:
                    st.info(f"Extra Parts ({validation_result['extra_parts_count']}):")
                    st.text("\n".join(validation_result['extra_parts_list'][:10]))
                    if len(validation_result['extra_parts_list']) > 10:
                        st.text(f"... and {len(validation_result['extra_parts_list']) - 10} more")
    
    def display_analysis_interface(self):
        """Main analysis interface for users"""
        st.subheader("📈 Inventory Analysis Results")
        # Get data
        pfep_data = self.persistence.load_data_from_session_state('persistent_pfep_data')
        inventory_data = self.persistence.load_data_from_session_state('persistent_inventory_data')
    
        if not pfep_data or not inventory_data:
            st.error("❌ Required data not available. Please upload PFEP and Inventory data first.")
            return
        # Get tolerance from admin settings
        tolerance = st.session_state.get('admin_tolerance', 30)
        st.info(f"📐 Analysis Tolerance: ±{tolerance}% (Set by Admin)")
    
        # Check if analysis needs to be performed or updated
        analysis_data = self.persistence.load_data_from_session_state('persistent_analysis_results')
        last_tolerance = st.session_state.get('last_analysis_tolerance', None)
    
        # Auto re-analyze if tolerance changed
        if not analysis_data or last_tolerance != tolerance:
            st.info(f"🔄 {'Re-analyzing' if analysis_data else 'Analyzing'} with tolerance ±{tolerance}%...")
            with st.spinner("Processing inventory analysis..."):
                analysis_results = self.analyzer.analyze_inventory(
                    pfep_data, 
                    inventory_data, 
                    tolerance=tolerance
                )
            if analysis_results:
                self.persistence.save_data_to_session_state('persistent_analysis_results', analysis_results)
                st.session_state.last_analysis_tolerance = tolerance
                st.success("✅ Analysis completed successfully!")
                st.rerun()
            else:
                st.error("❌ Analysis failed. Please check your data.")
                return
        # Display results
        self.display_analysis_results()

def display_comprehensive_analysis(self, analysis_results):
    """Display comprehensive analysis results with enhanced features"""
    st.success(f"✅ Analysis Complete: {len(analysis_results)} parts analyzed")
    
    # Summary metrics with better styling
    self.display_enhanced_summary_metrics(analysis_results)
    
    # Enhanced charts and visualizations
    self.display_enhanced_analysis_charts(analysis_results)
    
    # Improved detailed data tables
    self.display_enhanced_detailed_tables(analysis_results)
    
    # Advanced export options
    self.display_enhanced_export_options(analysis_results)

def display_enhanced_summary_metrics(self, analysis_results):
    """Enhanced summary metrics dashboard"""
    st.header("📊 Executive Summary Dashboard")
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .status-normal { background: linear-gradient(135deg, #4CAF50, #45a049); }
    .status-excess { background: linear-gradient(135deg, #2196F3, #1976D2); }
    .status-short { background: linear-gradient(135deg, #F44336, #D32F2F); }
    .status-total { background: linear-gradient(135deg, #FF9800, #F57C00); }
    
    .metric-card .metric-value { color: white; font-weight: bold; }
    .metric-card .metric-label { color: #f0f0f0; }
    
    .highlight-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    df = pd.DataFrame(analysis_results)
    
    # Calculate enhanced metrics
    total_parts = len(analysis_results)
    total_stock_value = df['Stock_Value'].sum() if 'Stock_Value' in df.columns else 0
    
    # Status distribution
    status_counts = Counter(df['Status'] if 'Status' in df.columns else df.get('INVENTORY REMARK STATUS', []))
    
    # Calculate financial impact
    short_impact = sum(item.get('VALUE(Unit Price* Short/Excess Inventory)', 0) 
                      for item in analysis_results 
                      if item.get('Status') == 'Short Inventory')
    
    excess_impact = sum(item.get('VALUE(Unit Price* Short/Excess Inventory)', 0) 
                       for item in analysis_results 
                       if item.get('Status') == 'Excess Inventory')
    
    # Display key performance indicators
    st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
    st.markdown(f"""
    ### 🎯 Key Performance Indicators
    - **Total Parts Analyzed**: {total_parts:,}
    - **Total Inventory Value**: ₹{total_stock_value:,.0f}
    - **Short Inventory Impact**: ₹{abs(short_impact):,.0f}
    - **Excess Inventory Impact**: ₹{excess_impact:,.0f}
    - **Net Financial Impact**: ₹{abs(short_impact) + excess_impact:,.0f}
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced metric cards with your original structure
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate values for each status
    summary_data = {}
    for status in status_counts:
        status_data = df[df['Status'] == status] if 'Status' in df.columns else df[df['INVENTORY REMARK STATUS'] == status]
        summary_data[status] = {
            'count': status_counts[status],
            'value': status_data['Stock_Value'].sum() if 'Stock_Value' in status_data.columns else 0
        }
    
    with col1:
        st.markdown('<div class="metric-card status-normal">', unsafe_allow_html=True)
        st.metric(
            label="🟢 Within Norms",
            value=f"{summary_data.get('Within Norms', {'count': 0})['count']} parts",
            delta=f"₹{summary_data.get('Within Norms', {'value': 0})['value']:,.0f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card status-excess">', unsafe_allow_html=True)
        st.metric(
            label="🔵 Excess Inventory",
            value=f"{summary_data.get('Excess Inventory', {'count': 0})['count']} parts",
            delta=f"₹{summary_data.get('Excess Inventory', {'value': 0})['value']:,.0f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card status-short">', unsafe_allow_html=True)
        st.metric(
            label="🔴 Short Inventory",
            value=f"{summary_data.get('Short Inventory', {'count': 0})['count']} parts",
            delta=f"₹{summary_data.get('Short Inventory', {'value': 0})['value']:,.0f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card status-total">', unsafe_allow_html=True)
        st.metric(
            label="📊 Total Value",
            value=f"{total_parts} parts",
            delta=f"₹{total_stock_value:,.0f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)

def display_enhanced_vendor_summary(self, analysis_results):
    """Enhanced vendor summary with better analytics"""
    st.header("🏢 Vendor Performance Analysis")
    
    df = pd.DataFrame(analysis_results)
    
    if 'Vendor' not in df.columns and 'Vendor Name' not in df.columns:
        st.warning("Vendor information not available in analysis data.")
        return
    
    vendor_col = 'Vendor' if 'Vendor' in df.columns else 'Vendor Name'
    
    # Calculate vendor metrics
    vendor_summary = {}
    for vendor in df[vendor_col].unique():
        vendor_data = df[df[vendor_col] == vendor]
        
        vendor_summary[vendor] = {
            'total_parts': len(vendor_data),
            'total_value': vendor_data['Stock_Value'].sum() if 'Stock_Value' in vendor_data.columns else 0,
            'short_parts': len(vendor_data[vendor_data['Status'] == 'Short Inventory']),
            'excess_parts': len(vendor_data[vendor_data['Status'] == 'Excess Inventory']),
            'normal_parts': len(vendor_data[vendor_data['Status'] == 'Within Norms']),
            'short_value': vendor_data[vendor_data['Status'] == 'Short Inventory']['Stock_Value'].sum(),
            'excess_value': vendor_data[vendor_data['Status'] == 'Excess Inventory']['Stock_Value'].sum(),
        }
    
    # Create enhanced vendor dataframe
    vendor_df = pd.DataFrame([
        {
            'Vendor': vendor,
            'Total Parts': data['total_parts'],
            'Short Inventory': data['short_parts'],
            'Excess Inventory': data['excess_parts'],
            'Within Norms': data['normal_parts'],
            'Total Value (₹)': f"₹{data['total_value']:,.0f}",
            'Performance Score': round((data['normal_parts'] / data['total_parts']) * 100, 1) if data['total_parts'] > 0 else 0
        }
        for vendor, data in vendor_summary.items()
    ])
    
    # Add color coding for performance
    def color_performance(val):
        if isinstance(val, str) and val.endswith('%'):
            score = float(val.replace('%', ''))
            if score >= 80:
                return 'background-color: #4CAF50; color: white'
            elif score >= 60:
                return 'background-color: #FF9800; color: white'
            else:
                return 'background-color: #F44336; color: white'
        return ''
    
    # Display vendor table with styling
    st.dataframe(
        vendor_df.style.applymap(color_performance, subset=['Performance Score']),
        use_container_width=True,
        hide_index=True
    )
    
    # Vendor performance chart
    fig = px.bar(
        vendor_df.head(10),
        x='Vendor',
        y='Performance Score',
        title="Top 10 Vendor Performance Scores",
        color='Performance Score',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig, use_container_width=True)

def create_enhanced_top_parts_chart(self, processed_data, status_filter, color, key, top_n=10):
    """Enhanced top parts chart with better visualization"""
    filtered_data = [
        item for item in processed_data 
        if item.get('Status') == status_filter or item.get('INVENTORY REMARK STATUS') == status_filter
    ]
    
    if not filtered_data:
        st.info(f"No {status_filter} parts found.")
        return
    
    # Sort by stock value
    top_parts = sorted(
        filtered_data,
        key=lambda x: x.get('Stock_Value', 0),
        reverse=True
    )[:top_n]
    
    # Create enhanced chart
    labels = [f"{item['PART NO']}<br>{item.get('PART DESCRIPTION', '')[:30]}..." for item in top_parts]
    values = [item.get('Stock_Value', 0) for item in top_parts]
    variance_values = [item.get('VALUE(Unit Price* Short/Excess Inventory)', 0) for item in top_parts]
    
    fig = go.Figure()
    
    # Add stock value bars
    fig.add_trace(go.Bar(
        name='Stock Value',
        x=labels,
        y=values,
        marker_color=color,
        text=[f"₹{v:,.0f}" for v in values],
        textposition='auto',
    ))
    
    # Add variance line
    fig.add_trace(go.Scatter(
        name='Variance Impact',
        x=labels,
        y=variance_values,
        mode='lines+markers',
        line=dict(color='red', width=2),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} {status_filter} Parts Analysis",
        xaxis_title="Parts",
        yaxis_title="Stock Value (₹)",
        yaxis2=dict(
            title="Variance Impact (₹)",
            overlaying='y',
            side='right'
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True, key=key)

def display_advanced_filtering_options(self, analysis_results):
    """Advanced filtering options for better data exploration"""
    st.sidebar.header("🔍 Advanced Filters")
    
    df = pd.DataFrame(analysis_results)
    
    # Value range filter
    if 'Stock_Value' in df.columns:
        min_value, max_value = st.sidebar.slider(
            "Stock Value Range (₹)",
            min_value=float(df['Stock_Value'].min()),
            max_value=float(df['Stock_Value'].max()),
            value=(float(df['Stock_Value'].min()), float(df['Stock_Value'].max())),
            format="₹%.0f"
        )
        st.session_state.value_filter = (min_value, max_value)
    
    # Quantity range filter
    if 'Current Inventory-QTY' in df.columns:
        min_qty, max_qty = st.sidebar.slider(
            "Quantity Range",
            min_value=int(df['Current Inventory-QTY'].min()),
            max_value=int(df['Current Inventory-QTY'].max()),
            value=(int(df['Current Inventory-QTY'].min()), int(df['Current Inventory-QTY'].max()))
        )
        st.session_state.qty_filter = (min_qty, max_qty)
    
    # Multi-select filters
    if 'Vendor' in df.columns or 'Vendor Name' in df.columns:
        vendor_col = 'Vendor' if 'Vendor' in df.columns else 'Vendor Name'
        selected_vendors = st.sidebar.multiselect(
            "Select Vendors",
            options=sorted(df[vendor_col].unique()),
            default=sorted(df[vendor_col].unique())
        )
        st.session_state.vendor_filter = selected_vendors
    
    # Critical parts filter
    critical_threshold = st.sidebar.number_input(
        "Critical Value Threshold (₹)",
        min_value=0,
        value=100000,
        step=10000
    )
    st.session_state.critical_threshold = critical_threshold
    
    return True

def apply_advanced_filters(self, df):
    """Apply advanced filters to dataframe"""
    filtered_df = df.copy()
    
    # Apply value filter
    if hasattr(st.session_state, 'value_filter') and 'Stock_Value' in df.columns:
        min_val, max_val = st.session_state.value_filter
        filtered_df = filtered_df[
            (filtered_df['Stock_Value'] >= min_val) & 
            (filtered_df['Stock_Value'] <= max_val)
        ]
    
    # Apply quantity filter
    if hasattr(st.session_state, 'qty_filter') and 'Current Inventory-QTY' in df.columns:
        min_qty, max_qty = st.session_state.qty_filter
        filtered_df = filtered_df[
            (filtered_df['Current Inventory-QTY'] >= min_qty) & 
            (filtered_df['Current Inventory-QTY'] <= max_qty)
        ]
    
    # Apply vendor filter
    if hasattr(st.session_state, 'vendor_filter'):
        vendor_col = 'Vendor' if 'Vendor' in df.columns else 'Vendor Name'
        if vendor_col in df.columns:
            filtered_df = filtered_df[filtered_df[vendor_col].isin(st.session_state.vendor_filter)]
    
    return filtered_df

def display_actionable_insights(self, analysis_results):
    """Display actionable insights and recommendations"""
    st.header("💡 Actionable Insights & Recommendations")
    
    df = pd.DataFrame(analysis_results)
    
    insights = []
    
    # Critical shortages
    critical_short = df[
        (df['Status'] == 'Short Inventory') & 
        (df['Stock_Value'] > st.session_state.get('critical_threshold', 100000))
    ]
    if not critical_short.empty:
        insights.append({
            'type': 'critical',
            'icon': '🚨',
            'title': 'Critical Shortages Detected',
            'message': f"{len(critical_short)} high-value parts are critically short. Immediate procurement required.",
            'action': 'Review procurement pipeline and expedite orders'
        })
    
    # Excess inventory opportunities
    excess_high = df[
        (df['Status'] == 'Excess Inventory') & 
        (df['Stock_Value'] > 50000)
    ]
    if not excess_high.empty:
        insights.append({
            'type': 'opportunity',
            'icon': '💰',
            'title': 'Cash Flow Optimization Opportunity',
            'message': f"₹{excess_high['Stock_Value'].sum():,.0f} tied up in excess inventory.",
            'action': 'Consider liquidation or redistribution strategies'
        })
    
    # Vendor performance issues
    vendor_col = 'Vendor' if 'Vendor' in df.columns else 'Vendor Name'
    if vendor_col in df.columns:
        vendor_performance = df.groupby(vendor_col).agg({
            'Status': lambda x: (x == 'Within Norms').mean()
        }).reset_index()
        
        poor_vendors = vendor_performance[vendor_performance['Status'] < 0.5]
        if not poor_vendors.empty:
            insights.append({
                'type': 'warning',
                'icon': '⚠️',
                'title': 'Vendor Performance Issues',
                'message': f"{len(poor_vendors)} vendors have <50% parts within norms.",
                'action': 'Schedule vendor performance reviews'
            })
    
    # Display insights
    for insight in insights:
        if insight['type'] == 'critical':
            st.error(f"{insight['icon']} **{insight['title']}**: {insight['message']}")
        elif insight['type'] == 'warning':
            st.warning(f"{insight['icon']} **{insight['title']}**: {insight['message']}")
        else:
            st.info(f"{insight['icon']} **{insight['title']}**: {insight['message']}")
        
        st.markdown(f"**Recommended Action**: {insight['action']}")
        st.markdown("---")
def display_trend_analysis(self, analysis_results):
    """Display trend analysis and forecasting"""
    st.header("📈 Trend Analysis & Forecasting")
    
    df = pd.DataFrame(analysis_results)
    
    # Create trend analysis tabs
    tab1, tab2, tab3 = st.tabs(["📊 Status Trends", "💹 Value Trends", "🔮 Forecasting"])
    
    with tab1:
        # Status distribution over time (if timestamp data available)
        status_counts = df['Status'].value_counts()
        
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Current Inventory Status Distribution",
            color_discrete_map={
                'Within Norms': '#4CAF50',
                'Excess Inventory': '#2196F3',
                'Short Inventory': '#F44336'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Status by category if available
        if 'Category' in df.columns or 'PART CATEGORY' in df.columns:
            category_col = 'Category' if 'Category' in df.columns else 'PART CATEGORY'
            status_category = df.groupby([category_col, 'Status']).size().unstack(fill_value=0)
            
            fig = px.bar(
                status_category,
                title="Status Distribution by Category",
                color_discrete_map={
                    'Within Norms': '#4CAF50',
                    'Excess Inventory': '#2196F3',
                    'Short Inventory': '#F44336'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Value distribution analysis
        value_ranges = pd.cut(df['Stock_Value'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        value_status = pd.crosstab(value_ranges, df['Status'])
        
        fig = px.bar(
            value_status,
            title="Status Distribution by Value Range",
            color_discrete_map={
                'Within Norms': '#4CAF50',
                'Excess Inventory': '#2196F3',
                'Short Inventory': '#F44336'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top value contributors
        top_value_parts = df.nlargest(20, 'Stock_Value')
        fig = px.scatter(
            top_value_parts,
            x='Current Inventory-QTY',
            y='Stock_Value',
            color='Status',
            size='Stock_Value',
            hover_data=['PART NO', 'PART DESCRIPTION'],
            title="Top 20 Parts by Value - Quantity vs Value Analysis"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔮 Predictive Insights")
        
        # Calculate reorder predictions
        reorder_candidates = df[
            (df['Status'] == 'Within Norms') & 
            (df['Current Inventory-QTY'] <= df['MIN QTY REQUIRED'] * 1.2)
        ]
        
        if not reorder_candidates.empty:
            st.warning(f"📋 **Reorder Alert**: {len(reorder_candidates)} parts may need reordering soon")
            
            # Display reorder table
            reorder_display = reorder_candidates[['PART NO', 'PART DESCRIPTION', 'Current Inventory-QTY', 
                                                'MIN QTY REQUIRED', 'Stock_Value']].copy()
            reorder_display['Days to Reorder'] = np.random.randint(5, 30, len(reorder_display))  # Simulated
            
            st.dataframe(reorder_display, use_container_width=True)
        
        # Seasonal analysis placeholder
        st.info("📊 **Seasonal Analysis**: Historical data integration required for advanced forecasting")

def display_executive_dashboard(self, analysis_results):
    """Executive level dashboard with KPIs"""
    st.header("👔 Executive Dashboard")
    
    df = pd.DataFrame(analysis_results)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_value = df['Stock_Value'].sum()
        st.metric(
            "Total Inventory Value",
            f"₹{total_value:,.0f}",
            delta=f"Analyzed: {len(df)} parts"
        )
    
    with col2:
        efficiency = (df['Status'] == 'Within Norms').mean() * 100
        st.metric(
            "Inventory Efficiency",
            f"{efficiency:.1f}%",
            delta=f"{'Good' if efficiency > 70 else 'Needs Improvement'}"
        )
    
    with col3:
        excess_value = df[df['Status'] == 'Excess Inventory']['Stock_Value'].sum()
        st.metric(
            "Excess Inventory",
            f"₹{excess_value:,.0f}",
            delta=f"{(df['Status'] == 'Excess Inventory').sum()} parts"
        )
    
    with col4:
        short_impact = abs(df[df['Status'] == 'Short Inventory']['VALUE(Unit Price* Short/Excess Inventory)'].sum())
        st.metric(
            "Shortage Impact",
            f"₹{short_impact:,.0f}",
            delta=f"{(df['Status'] == 'Short Inventory').sum()} parts"
        )
    
    # Risk matrix
    st.subheader("🎯 Risk Assessment Matrix")
    
    # Create risk categories
    df['Risk_Level'] = 'Low'
    df.loc[(df['Stock_Value'] > 100000) & (df['Status'] != 'Within Norms'), 'Risk_Level'] = 'High'
    df.loc[(df['Stock_Value'] > 50000) & (df['Stock_Value'] <= 100000) & (df['Status'] != 'Within Norms'), 'Risk_Level'] = 'Medium'
    
    risk_matrix = df.groupby(['Risk_Level', 'Status']).agg({
        'Stock_Value': 'sum',
        'PART NO': 'count'
    }).reset_index()
    
    fig = px.sunburst(
        risk_matrix,
        path=['Risk_Level', 'Status'],
        values='Stock_Value',
        title="Risk Assessment by Value Impact"
    )
    st.plotly_chart(fig, use_container_width=True)

def display_export_options(self, analysis_results):
    """Enhanced export options"""
    st.header("📥 Export & Reporting Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Full Analysis", type="primary"):
            self.export_comprehensive_report(analysis_results)
    
    with col2:
        if st.button("🚨 Export Critical Items Only"):
            self.export_critical_items(analysis_results)
    
    with col3:
        if st.button("📈 Export Executive Summary"):
            self.export_executive_summary(analysis_results)
    
    # Export format options
    st.subheader("Export Format Options")
    export_format = st.selectbox(
        "Select Export Format",
        ["Excel (.xlsx)", "CSV (.csv)", "PDF Report", "PowerPoint Summary"]
    )
    
    if st.button("🎯 Custom Export"):
        self.export_custom_format(analysis_results, export_format)

def export_comprehensive_report(self, analysis_results):
    """Export comprehensive analysis report"""
    try:
        df = pd.DataFrame(analysis_results)
        
        # Create Excel writer
        from io import BytesIO
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Main analysis sheet
            df.to_excel(writer, sheet_name='Full Analysis', index=False)
            
            # Summary sheet
            summary_data = {
                'Status': df['Status'].value_counts().index.tolist(),
                'Count': df['Status'].value_counts().values.tolist(),
                'Total Value': [df[df['Status'] == status]['Stock_Value'].sum() 
                               for status in df['Status'].value_counts().index]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            # Critical items sheet
            critical_items = df[df['Stock_Value'] > 100000]
            critical_items.to_excel(writer, sheet_name='Critical Items', index=False)
        
        # Download button
        st.download_button(
            label="📥 Download Comprehensive Report",
            data=output.getvalue(),
            file_name=f"inventory_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.success("✅ Comprehensive report prepared for download!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def export_critical_items(self, analysis_results):
    """Export only critical items"""
    try:
        df = pd.DataFrame(analysis_results)
        
        # Filter critical items
        critical_items = df[
            (df['Status'] != 'Within Norms') & 
            (df['Stock_Value'] > st.session_state.get('critical_threshold', 100000))
        ]
        
        if critical_items.empty:
            st.warning("No critical items found based on current criteria.")
            return
        
        # Create CSV
        csv = critical_items.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Critical Items Report",
            data=csv,
            file_name=f"critical_items_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.success(f"✅ Critical items report prepared! ({len(critical_items)} items)")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def export_executive_summary(self, analysis_results):
    """Export executive summary"""
    try:
        df = pd.DataFrame(analysis_results)
        
        # Create executive summary data
        summary = {
            'Metric': [
                'Total Parts Analyzed',
                'Total Inventory Value (₹)',
                'Parts Within Norms',
                'Excess Inventory Parts',
                'Short Inventory Parts',
                'Inventory Efficiency (%)',
                'Excess Value (₹)',
                'Shortage Impact (₹)'
            ],
            'Value': [
                len(df),
                f"₹{df['Stock_Value'].sum():,.0f}",
                (df['Status'] == 'Within Norms').sum(),
                (df['Status'] == 'Excess Inventory').sum(),
                (df['Status'] == 'Short Inventory').sum(),
                f"{(df['Status'] == 'Within Norms').mean() * 100:.1f}%",
                f"₹{df[df['Status'] == 'Excess Inventory']['Stock_Value'].sum():,.0f}",
                f"₹{abs(df[df['Status'] == 'Short Inventory']['VALUE(Unit Price* Short/Excess Inventory)'].sum()):,.0f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary)
        csv = summary_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Executive Summary",
            data=csv,
            file_name=f"executive_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        st.success("✅ Executive summary prepared for download!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def export_custom_format(self, analysis_results, format_type):
    """Export in custom format"""
    try:
        df = pd.DataFrame(analysis_results)
        
        if format_type == "Excel (.xlsx)":
            self.export_comprehensive_report(analysis_results)
        elif format_type == "CSV (.csv)":
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"inventory_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        elif format_type == "PDF Report":
            st.info("📄 PDF export functionality requires additional setup. Using CSV format instead.")
            self.export_custom_format(analysis_results, "CSV (.csv)")
        elif format_type == "PowerPoint Summary":
            st.info("📊 PowerPoint export functionality requires additional setup. Using Excel format instead.")
            self.export_comprehensive_report(analysis_results)
        
        st.success(f"✅ Export completed in {format_type} format!")
        
    except Exception as e:
        st.error(f"❌ Export failed: {str(e)}")

def display_help_and_documentation(self):
    """Display help and documentation"""
    st.header("❓ Help & Documentation")
    
    with st.expander("📖 Understanding Analysis Results"):
        st.markdown("""
        ### Status Categories:
        - **🟢 Within Norms**: Inventory levels are optimal
        - **🔵 Excess Inventory**: Stock levels exceed requirements
        - **🔴 Short Inventory**: Stock levels are below minimum requirements
        
        ### Key Metrics:
        - **Stock Value**: Total monetary value of current inventory
        - **Variance Impact**: Financial impact of excess/shortage
        - **Performance Score**: Percentage of parts within norms
        """)
    
    with st.expander("🔧 Advanced Features"):
        st.markdown("""
        ### Filtering Options:
        - Use sidebar filters to focus on specific value ranges
        - Filter by vendor performance
        - Set critical value thresholds
        
        ### Export Options:
        - **Full Analysis**: Complete detailed report
        - **Critical Items**: High-value problem items only
        - **Executive Summary**: Key metrics for management
        """)
    
    with st.expander("💡 Best Practices"):
        st.markdown("""
        ### Optimization Tips:
        1. **Regular Monitoring**: Run analysis weekly/monthly
        2. **Vendor Performance**: Track vendor consistency
        3. **Critical Thresholds**: Adjust based on business needs
        4. **Action Items**: Follow up on recommendations
        5. **Trend Analysis**: Monitor patterns over time
        """)

def display_analysis_results(self):
    """Main method to display all analysis results"""
    analysis_results = self.persistence.load_data_from_session_state('persistent_analysis_results')
    
    if not analysis_results:
        st.error("❌ No analysis results available.")
        return
    
    # Display advanced filtering options
    self.display_advanced_filtering_options(analysis_results)
    
    # Apply filters to data
    df = pd.DataFrame(analysis_results)
    filtered_df = self.apply_advanced_filters(df)
    filtered_results = filtered_df.to_dict('records')
    
    # Display main dashboard
    self.display_comprehensive_analysis(filtered_results)
    
    # Additional analysis sections
    st.markdown("---")
    self.display_trend_analysis(filtered_results)
    
    st.markdown("---")
    self.display_executive_dashboard(filtered_results)
    
    st.markdown("---")
    self.display_actionable_insights(filtered_results)
    
    st.markdown("---")
    self.display_export_options(filtered_results)
    
    st.markdown("---")
    self.display_help_and_documentation()
def display_actionable_insights(self, analysis_results):
    """Display actionable insights and recommendations"""
    st.header("💡 Actionable Insights & Recommendations")
    
    df = pd.DataFrame(analysis_results)
    
    # Create insights tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Immediate Actions", "💰 Cost Optimization", "📊 Performance", "🔄 Process Improvement"])
    
    with tab1:
        st.subheader("🚨 Immediate Action Required")
        
        # Critical shortages
        critical_shortages = df[
            (df['Status'] == 'Short Inventory') & 
            (df['Stock_Value'] > 50000)
        ].sort_values('Stock_Value', ascending=False)
        
        if not critical_shortages.empty:
            st.error(f"🔴 **URGENT**: {len(critical_shortages)} high-value parts critically short!")
            
            for idx, part in critical_shortages.head(5).iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 2])
                    with col1:
                        st.write(f"**{part['PART NO']}** - {part['PART DESCRIPTION'][:50]}...")
                    with col2:
                        st.write(f"Value: ₹{part['Stock_Value']:,.0f}")
                    with col3:
                        shortage = part['MIN QTY REQUIRED'] - part['Current Inventory-QTY']
                        st.write(f"Need: {shortage} units")
        
        # Excess inventory actions
        excess_items = df[
            (df['Status'] == 'Excess Inventory') & 
            (df['Stock_Value'] > 100000)
        ].sort_values('Stock_Value', ascending=False)
        
        if not excess_items.empty:
            st.warning(f"🟡 **Consider**: {len(excess_items)} high-value excess items for reallocation")
    
    with tab2:
        st.subheader("💰 Cost Optimization Opportunities")
        
        # Calculate potential savings
        excess_value = df[df['Status'] == 'Excess Inventory']['Stock_Value'].sum()
        potential_savings = excess_value * 0.1  # Assume 10% carrying cost
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Potential Annual Savings",
                f"₹{potential_savings:,.0f}",
                help="Based on 10% carrying cost reduction"
            )
        
        with col2:
            freed_capital = excess_value * 0.7  # 70% could be freed up
            st.metric(
                "Capital That Could Be Freed",
                f"₹{freed_capital:,.0f}",
                help="From excess inventory optimization"
            )
        
        # Top optimization candidates
        st.subheader("🎯 Top Optimization Candidates")
        optimization_candidates = df[df['Status'] == 'Excess Inventory'].nlargest(10, 'Stock_Value')
        
        if not optimization_candidates.empty:
            opt_display = optimization_candidates[['PART NO', 'PART DESCRIPTION', 'Current Inventory-QTY', 
                                                 'MIN QTY REQUIRED', 'Stock_Value']].copy()
            opt_display['Excess Qty'] = opt_display['Current Inventory-QTY'] - opt_display['MIN QTY REQUIRED']
            opt_display['Optimization Potential'] = opt_display['Excess Qty'] * (opt_display['Stock_Value'] / opt_display['Current Inventory-QTY'])
            
            st.dataframe(opt_display, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Performance Analysis")
        
        # Vendor performance insights
        if 'VENDOR' in df.columns:
            vendor_performance = df.groupby('VENDOR').agg({
                'Status': lambda x: (x == 'Within Norms').mean() * 100,
                'Stock_Value': 'sum',
                'PART NO': 'count'
            }).round(2)
            vendor_performance.columns = ['Performance %', 'Total Value', 'Part Count']
            vendor_performance = vendor_performance.sort_values('Performance %', ascending=False)
            
            st.subheader("🏆 Vendor Performance Ranking")
            st.dataframe(vendor_performance, use_container_width=True)
        
        # Category performance
        if 'Category' in df.columns or 'PART CATEGORY' in df.columns:
            category_col = 'Category' if 'Category' in df.columns else 'PART CATEGORY'
            category_performance = df.groupby(category_col).agg({
                'Status': lambda x: (x == 'Within Norms').mean() * 100,
                'Stock_Value': 'sum'
            }).round(2)
            category_performance.columns = ['Performance %', 'Total Value']
            
            fig = px.scatter(
                category_performance.reset_index(),
                x='Performance %',
                y='Total Value',
                size='Total Value',
                hover_data=[category_col],
                title="Category Performance vs Value Analysis"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🔄 Process Improvement Recommendations")
        
        improvement_score = (df['Status'] == 'Within Norms').mean() * 100
        
        recommendations = []
        
        if improvement_score < 60:
            recommendations.extend([
                "🔴 **Critical**: Implement ABC analysis for better inventory classification",
                "🔴 **Critical**: Review and update MIN/MAX thresholds quarterly",
                "🔴 **Critical**: Establish vendor performance monitoring system"
            ])
        elif improvement_score < 80:
            recommendations.extend([
                "🟡 **Important**: Consider implementing JIT for fast-moving items",
                "🟡 **Important**: Set up automated reorder alerts",
                "🟡 **Important**: Review slow-moving inventory monthly"
            ])
        else:
            recommendations.extend([
                "🟢 **Good**: Maintain current practices with minor optimizations",
                "🟢 **Good**: Consider advanced forecasting methods",
                "🟢 **Good**: Implement continuous improvement processes"
            ])
        
        # Display recommendations
        for rec in recommendations:
            st.markdown(rec)
        
        # Process improvement metrics
        st.subheader("📈 Suggested KPIs to Track")
        kpi_col1, kpi_col2 = st.columns(2)
        
        with kpi_col1:
            st.markdown("""
            **Operational KPIs:**
            - Inventory Turnover Ratio
            - Stockout Frequency
            - Lead Time Variability
            - Carrying Cost %
            """)
        
        with kpi_col2:
            st.markdown("""
            **Financial KPIs:**
            - Inventory-to-Sales Ratio
            - Obsolete Inventory %
            - Working Capital Efficiency
            - Cost of Stockouts
            """)

def display_advanced_filtering_options(self, analysis_results):
    """Display advanced filtering options in sidebar"""
    st.sidebar.header("🔍 Advanced Filters")
    
    df = pd.DataFrame(analysis_results)
    
    # Value range filter
    min_value = float(df['Stock_Value'].min())
    max_value = float(df['Stock_Value'].max())
    
    value_range = st.sidebar.slider(
        "Stock Value Range (₹)",
        min_value=min_value,
        max_value=max_value,
        value=(min_value, max_value),
        format="₹%.0f"
    )
    
    # Status filter
    status_options = df['Status'].unique().tolist()
    selected_statuses = st.sidebar.multiselect(
        "Filter by Status",
        options=status_options,
        default=status_options
    )
    
    # Category filter (if available)
    category_col = None
    if 'Category' in df.columns:
        category_col = 'Category'
    elif 'PART CATEGORY' in df.columns:
        category_col = 'PART CATEGORY'
    
    selected_categories = None
    if category_col:
        categories = df[category_col].unique().tolist()
        selected_categories = st.sidebar.multiselect(
            f"Filter by {category_col}",
            options=categories,
            default=categories
        )
    
    # Vendor filter (if available)
    selected_vendors = None
    if 'VENDOR' in df.columns:
        vendors = df['VENDOR'].unique().tolist()
        selected_vendors = st.sidebar.multiselect(
            "Filter by Vendor",
            options=vendors,
            default=vendors
        )
    
    # Critical threshold setting
    critical_threshold = st.sidebar.number_input(
        "Critical Value Threshold (₹)",
        min_value=0,
        value=100000,
        step=10000,
        help="Parts above this value are considered critical"
    )
    
    # Store filter values in session state
    st.session_state.filter_value_range = value_range
    st.session_state.filter_statuses = selected_statuses
    st.session_state.filter_categories = selected_categories
    st.session_state.filter_vendors = selected_vendors
    st.session_state.critical_threshold = critical_threshold

def apply_advanced_filters(self, df):
    """Apply advanced filters to the dataframe"""
    filtered_df = df.copy()
    
    # Apply value range filter
    if hasattr(st.session_state, 'filter_value_range'):
        min_val, max_val = st.session_state.filter_value_range
        filtered_df = filtered_df[
            (filtered_df['Stock_Value'] >= min_val) & 
            (filtered_df['Stock_Value'] <= max_val)
        ]
    
    # Apply status filter
    if hasattr(st.session_state, 'filter_statuses') and st.session_state.filter_statuses:
        filtered_df = filtered_df[filtered_df['Status'].isin(st.session_state.filter_statuses)]
    
    # Apply category filter
    if hasattr(st.session_state, 'filter_categories') and st.session_state.filter_categories:
        category_col = 'Category' if 'Category' in filtered_df.columns else 'PART CATEGORY'
        if category_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[category_col].isin(st.session_state.filter_categories)]
    
    # Apply vendor filter
    if hasattr(st.session_state, 'filter_vendors') and st.session_state.filter_vendors:
        if 'VENDOR' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['VENDOR'].isin(st.session_state.filter_vendors)]
    
    return filtered_df

def generate_analysis_summary(self, analysis_results):
    """Generate a comprehensive analysis summary"""
    df = pd.DataFrame(analysis_results)
    
    summary = {
        'total_parts': len(df),
        'total_value': df['Stock_Value'].sum(),
        'within_norms': (df['Status'] == 'Within Norms').sum(),
        'excess_inventory': (df['Status'] == 'Excess Inventory').sum(),
        'short_inventory': (df['Status'] == 'Short Inventory').sum(),
        'efficiency_rate': (df['Status'] == 'Within Norms').mean() * 100,
        'excess_value': df[df['Status'] == 'Excess Inventory']['Stock_Value'].sum(),
        'shortage_impact': abs(df[df['Status'] == 'Short Inventory']['VALUE(Unit Price* Short/Excess Inventory)'].sum()),
        'critical_items': len(df[df['Stock_Value'] > st.session_state.get('critical_threshold', 100000)]),
        'avg_stock_value': df['Stock_Value'].mean()
    }
    
    return summary

def main(self):
    """Main execution method for the inventory analyzer"""
    st.set_page_config(
        page_title="Advanced Inventory Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Main application logic
    if not st.session_state.analysis_complete:
        # Show file upload and analysis interface
        self.display_file_upload_interface()
    else:
        # Show analysis results
        self.display_analysis_results()
        
        # Option to analyze new file
        if st.sidebar.button("🔄 Analyze New File", type="secondary"):
            st.session_state.analysis_complete = False
            st.rerun()

# Alternative approach - if main should be a standalone function:
# Remove 'self' parameter from the main method definition and change it to:
def main():
    """Main execution method for the inventory analyzer"""
    st.set_page_config(
        page_title="Advanced Inventory Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Create analyzer instance here
    analyzer = InventoryAnalyzer()  # Replace with your actual class name
    
    # Main application logic
    if not st.session_state.analysis_complete:
        # Show file upload and analysis interface
        analyzer.display_file_upload_interface()
    else:
        # Show analysis results
        analyzer.display_analysis_results()
        
        # Option to analyze new file
        if st.sidebar.button("🔄 Analyze New File", type="secondary"):
            st.session_state.analysis_complete = False
            st.rerun()

# Then keep the if __name__ == "__main__": block as:
# Usage example and initialization
if __name__ == "__main__":
    analyzer = InventoryAnalyzer()
    analyzer.main()
