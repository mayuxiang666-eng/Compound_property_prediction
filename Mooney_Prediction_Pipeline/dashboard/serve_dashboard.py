# ============================================================================
# V3.6 Production Web Dashboard HTTP & Real MMS Data Server
# ============================================================================
# Serves the interactive web dashboard on localhost:8050
# Provides REST API /api/real_batches returning actual MMS production data.
# ============================================================================

import http.server
import json
import os
import socketserver
import pandas as pd

PORT = 8050
dashboard_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_root = os.path.abspath(os.path.join(dashboard_dir, '..'))


class RealDataDashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP Handler serving static assets and real MMS dataset API."""

    def do_GET(self):
        if self.path.startswith('/api/real_batches'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            csv_path = os.path.join(pipeline_root, 'reports', 'v36_explainable_production', 'time_series_calibration_stream.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)

                # Add batch_no within each OrderID
                df['Batch_No'] = df.groupby('OrderID').cumcount() + 1
                df['Total_In_Order'] = df.groupby('OrderID')['OrderID'].transform('count')

                # Replace NaNs
                df = df.fillna(0.0)

                records = df.to_dict(orient='records')
                self.wfile.write(json.dumps(records).encode('utf-8'))
            else:
                self.wfile.write(json.dumps([]).encode('utf-8'))
            return
        
        return super().do_GET()


def serve_dashboard():
    os.chdir(dashboard_dir)
    with socketserver.TCPServer(("", PORT), RealDataDashboardHandler) as httpd:
        print("=" * 80)
        print(f"  V3.6 REAL MMS PRODUCTION DASHBOARD LIVE AT:")
        print(f"  http://localhost:{PORT}/index.html")
        print("=" * 80)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")


if __name__ == '__main__':
    serve_dashboard()
