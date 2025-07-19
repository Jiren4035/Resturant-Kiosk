from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime, timedelta, date
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import random
import math
import numpy as np
from collections import defaultdict
import uuid
import re

app = Flask(__name__)
CORS(app)


@dataclass
class MenuItem:
    id: str
    name: str
    category: str
    price: float
    cost: float
    description: str
    ingredients: List[str]
    preparation_time: int
    popularity_score: float
    profit_margin: float
    dietary_tags: List[str]
    created_at: str
    is_available: bool = True


@dataclass
class InventoryItem:
    id: str
    name: str
    category: str
    current_stock: float
    unit: str
    reorder_level: float
    supplier: str
    cost_per_unit: float
    expiry_date: Optional[str]
    last_updated: str


@dataclass
class Order:
    id: str
    table_number: int
    items: List[Dict]
    total_amount: float
    status: str
    order_time: str
    estimated_ready_time: str
    customer_satisfaction: Optional[int]
    special_notes: str


@dataclass
class Staff:
    id: str
    name: str
    role: str
    hourly_rate: float
    efficiency_score: float
    availability: List[str]
    created_at: str


class MenuOptimizer:
    def __init__(self):
        self.seasonal_factors = {
            'spring': {'salads': 1.3, 'soups': 0.7, 'cold_drinks': 1.1},
            'summer': {'cold_drinks': 1.5, 'ice_cream': 1.4, 'salads': 1.2, 'hot_drinks': 0.6},
            'autumn': {'soups': 1.3, 'warm_drinks': 1.2, 'comfort_food': 1.2},
            'winter': {'soups': 1.4, 'hot_drinks': 1.3, 'comfort_food': 1.3, 'cold_drinks': 0.7}
        }

    def analyze_menu_performance(self, sales_data: List[Dict], menu_items: List[MenuItem]) -> Dict:
        analysis = {
            'top_performers': [],
            'underperformers': [],
            'profitability_insights': [],
            'seasonal_recommendations': [],
            'pricing_suggestions': []
        }

        item_metrics = {}
        for item in menu_items:
            sales = [order for order in sales_data if
                     any(oi['menu_item_id'] == item.id for oi in order.get('items', []))]
            total_quantity = sum(
                oi['quantity'] for order in sales for oi in order.get('items', []) if oi['menu_item_id'] == item.id)
            total_revenue = total_quantity * item.price
            total_cost = total_quantity * item.cost
            profit = total_revenue - total_cost

            item_metrics[item.id] = {
                'name': item.name,
                'total_sales': total_quantity,
                'revenue': total_revenue,
                'profit': profit,
                'profit_margin': item.profit_margin,
                'popularity_score': item.popularity_score
            }

        sorted_by_performance = sorted(item_metrics.items(),
                                       key=lambda x: x[1]['total_sales'] * x[1]['profit_margin'],
                                       reverse=True)

        analysis['top_performers'] = [
            {
                'item_name': metrics['name'],
                'performance_score': metrics['total_sales'] * metrics['profit_margin'],
                'recommendation': 'Promote this item - high sales and profitability'
            }
            for _, metrics in sorted_by_performance[:5]
        ]

        analysis['underperformers'] = [
            {
                'item_name': metrics['name'],
                'issues': self._diagnose_performance_issues(metrics),
                'recommendation': self._get_improvement_recommendation(metrics)
            }
            for _, metrics in sorted_by_performance[-3:] if metrics['total_sales'] < 10
        ]

        current_season = self._get_current_season()
        analysis['seasonal_recommendations'] = self._generate_seasonal_recommendations(current_season, menu_items)
        analysis['pricing_suggestions'] = self._analyze_pricing_opportunities(item_metrics)

        return analysis

    def predict_demand(self, menu_item_id: str, historical_data: List[Dict], days_ahead: int = 7) -> Dict:
        recent_sales = []
        for i in range(min(30, len(historical_data))):
            day_sales = sum(oi['quantity'] for order in historical_data[-(i + 1):]
                            for oi in order.get('items', []) if oi['menu_item_id'] == menu_item_id)
            recent_sales.append(day_sales)

        if not recent_sales:
            return {'predicted_demand': 0, 'confidence': 0, 'trend': 'no_data'}

        avg_sales = np.mean(recent_sales)
        trend = self._calculate_trend(recent_sales)
        seasonal_factor = self._get_seasonal_factor(menu_item_id)

        base_prediction = avg_sales * seasonal_factor
        trend_adjustment = trend * days_ahead * 0.1
        predicted_demand = max(0, base_prediction + trend_adjustment)

        confidence = min(100, len(recent_sales) * 3)

        return {
            'predicted_demand': round(predicted_demand, 1),
            'confidence': confidence,
            'trend': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable',
            'seasonal_factor': seasonal_factor
        }

    def optimize_menu_pricing(self, item: MenuItem, competitor_prices: Dict, demand_elasticity: float = -0.5) -> Dict:
        current_price = item.price
        optimal_price = current_price

        min_price = item.cost * 1.2
        max_price = current_price * 1.5

        avg_competitor_price = np.mean(list(competitor_prices.values())) if competitor_prices else current_price

        price_candidates = np.linspace(min_price, max_price, 20)
        best_score = 0

        for price in price_candidates:
            demand_change = demand_elasticity * ((price - current_price) / current_price)
            estimated_demand = item.popularity_score * (1 + demand_change)

            profit_per_unit = price - item.cost
            total_profit_score = estimated_demand * profit_per_unit

            competitive_score = 1.0 if abs(price - avg_competitor_price) < 2 else 0.8

            final_score = total_profit_score * competitive_score

            if final_score > best_score:
                best_score = final_score
                optimal_price = price

        price_change_percent = ((optimal_price - current_price) / current_price) * 100

        return {
            'current_price': current_price,
            'optimal_price': round(optimal_price, 2),
            'price_change_percent': round(price_change_percent, 1),
            'expected_impact': self._get_pricing_impact_description(price_change_percent),
            'recommendation': 'increase' if optimal_price > current_price else 'decrease' if optimal_price < current_price else 'maintain'
        }

    def _diagnose_performance_issues(self, metrics: Dict) -> List[str]:
        issues = []
        if metrics['total_sales'] < 5:
            issues.append('Low sales volume')
        if metrics['profit_margin'] < 0.3:
            issues.append('Low profit margin')
        if metrics['popularity_score'] < 0.5:
            issues.append('Low customer preference')
        return issues

    def _get_improvement_recommendation(self, metrics: Dict) -> str:
        if metrics['profit_margin'] < 0.3:
            return 'Consider reducing ingredient costs or increasing price'
        elif metrics['total_sales'] < 5:
            return 'Improve marketing or consider removing from menu'
        else:
            return 'Monitor performance and gather customer feedback'

    def _get_current_season(self) -> str:
        month = datetime.now().month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'

    def _generate_seasonal_recommendations(self, season: str, menu_items: List[MenuItem]) -> List[Dict]:
        recommendations = []
        seasonal_factors = self.seasonal_factors.get(season, {})

        for category, factor in seasonal_factors.items():
            if factor > 1.1:
                items = [item for item in menu_items if category.lower() in item.category.lower()]
                if items:
                    recommendations.append({
                        'action': 'promote',
                        'category': category,
                        'reason': f'{season.title()} season increases demand by {int((factor - 1) * 100)}%',
                        'items': [item.name for item in items[:3]]
                    })

        return recommendations

    def _calculate_trend(self, data: List[float]) -> float:
        if len(data) < 2:
            return 0
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        return coeffs[0]

    def _get_seasonal_factor(self, menu_item_id: str) -> float:
        return random.uniform(0.8, 1.3)

    def _get_pricing_impact_description(self, change_percent: float) -> str:
        if change_percent > 10:
            return 'Significant revenue increase expected, monitor customer response'
        elif change_percent > 5:
            return 'Moderate revenue increase expected'
        elif change_percent < -10:
            return 'Price reduction may increase volume significantly'
        elif change_percent < -5:
            return 'Moderate price reduction, expect volume increase'
        else:
            return 'Minor price adjustment, minimal impact expected'

    def _analyze_pricing_opportunities(self, item_metrics: Dict) -> List[Dict]:
        return []


class InventoryAI:
    def predict_inventory_needs(self, inventory_items: List[InventoryItem],
                                sales_data: List[Dict], days_ahead: int = 7) -> Dict:
        predictions = {}

        for item in inventory_items:
            usage_rate = self._calculate_usage_rate(item, sales_data)
            predicted_consumption = usage_rate * days_ahead
            safety_factor = self._calculate_safety_factor(item, sales_data)
            safety_stock = predicted_consumption * safety_factor
            total_needed = predicted_consumption + safety_stock
            current_stock = item.current_stock

            predictions[item.id] = {
                'item_name': item.name,
                'current_stock': current_stock,
                'predicted_consumption': round(predicted_consumption, 2),
                'safety_stock': round(safety_stock, 2),
                'total_needed': round(total_needed, 2),
                'reorder_needed': total_needed > current_stock,
                'reorder_quantity': max(0, round(total_needed - current_stock + item.reorder_level, 2)),
                'urgency': self._calculate_urgency(current_stock, usage_rate, item.reorder_level),
                'cost_impact': round((total_needed - current_stock) * item.cost_per_unit, 2)
            }

        return predictions

    def optimize_reorder_points(self, item: InventoryItem, historical_data: List[Dict]) -> Dict:
        usage_rates = self._get_historical_usage_rates(item, historical_data)

        if not usage_rates:
            return {
                'current_reorder_level': item.reorder_level,
                'optimal_reorder_level': item.reorder_level,
                'reasoning': 'Insufficient data for optimization'
            }

        avg_usage = np.mean(usage_rates)
        usage_std = np.std(usage_rates) if len(usage_rates) > 1 else avg_usage * 0.2
        lead_time = 3
        service_level = 1.96

        optimal_reorder_level = (avg_usage * lead_time) + (service_level * usage_std * math.sqrt(lead_time))

        return {
            'current_reorder_level': item.reorder_level,
            'optimal_reorder_level': round(optimal_reorder_level, 2),
            'potential_savings': self._calculate_potential_savings(item, optimal_reorder_level),
            'reasoning': f'Based on average usage of {avg_usage:.1f} {item.unit}/day and lead time of {lead_time} days'
        }

    def detect_waste_opportunities(self, inventory_items: List[InventoryItem],
                                   sales_data: List[Dict]) -> List[Dict]:
        opportunities = []

        for item in inventory_items:
            if item.expiry_date:
                expiry_date = datetime.fromisoformat(item.expiry_date)
                days_to_expiry = (expiry_date - datetime.now()).days

                if days_to_expiry <= 3 and item.current_stock > 0:
                    opportunities.append({
                        'type': 'expiry_risk',
                        'item_name': item.name,
                        'issue': f'{item.current_stock} {item.unit} expiring in {days_to_expiry} days',
                        'recommendation': 'Create promotional offer or use in daily specials',
                        'potential_loss': round(item.current_stock * item.cost_per_unit, 2),
                        'urgency': 'high' if days_to_expiry <= 1 else 'medium'
                    })

            usage_rate = self._calculate_usage_rate(item, sales_data)
            if usage_rate > 0:
                days_of_stock = item.current_stock / usage_rate
                if days_of_stock > 14:
                    opportunities.append({
                        'type': 'overstock',
                        'item_name': item.name,
                        'issue': f'{days_of_stock:.0f} days of stock on hand',
                        'recommendation': 'Reduce next order quantity or create promotions',
                        'excess_stock': round(item.current_stock - (usage_rate * 7), 2),
                        'urgency': 'low'
                    })

        return opportunities

    def _calculate_usage_rate(self, item: InventoryItem, sales_data: List[Dict]) -> float:
        total_usage = len(sales_data) * 0.1
        days = min(30, len(sales_data))
        return total_usage / max(1, days)

    def _calculate_safety_factor(self, item: InventoryItem, sales_data: List[Dict]) -> float:
        return 0.2

    def _calculate_urgency(self, current_stock: float, usage_rate: float, reorder_level: float) -> str:
        if usage_rate <= 0:
            return 'low'

        days_remaining = current_stock / usage_rate
        if days_remaining <= 2:
            return 'critical'
        elif days_remaining <= 5:
            return 'high'
        elif days_remaining <= 10:
            return 'medium'
        else:
            return 'low'

    def _get_historical_usage_rates(self, item: InventoryItem, historical_data: List[Dict]) -> List[float]:
        return [random.uniform(1, 10) for _ in range(min(30, len(historical_data)))]

    def _calculate_potential_savings(self, item: InventoryItem, optimal_level: float) -> float:
        current_carrying_cost = item.reorder_level * item.cost_per_unit * 0.1
        optimal_carrying_cost = optimal_level * item.cost_per_unit * 0.1
        return round(current_carrying_cost - optimal_carrying_cost, 2)


class StaffScheduler:
    def optimize_schedule(self, staff: List[Staff], predicted_demand: Dict,
                          days_ahead: int = 7) -> Dict:
        schedule = {}

        for day in range(days_ahead):
            date_str = (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d')
            day_name = (datetime.now() + timedelta(days=day)).strftime('%A').lower()

            predicted_customers = self._predict_daily_customers(day_name, predicted_demand)
            staff_requirements = self._calculate_staff_requirements(predicted_customers)
            assigned_staff = self._assign_staff(staff, staff_requirements, day_name)

            schedule[date_str] = {
                'predicted_customers': predicted_customers,
                'staff_requirements': staff_requirements,
                'assigned_staff': assigned_staff,
                'total_cost': sum(s['hours'] * s['hourly_rate'] for s in assigned_staff),
                'efficiency_score': self._calculate_schedule_efficiency(assigned_staff, staff_requirements)
            }

        return schedule

    def _predict_daily_customers(self, day_name: str, predicted_demand: Dict) -> int:
        base_customers = {
            'monday': 80, 'tuesday': 90, 'wednesday': 95, 'thursday': 100,
            'friday': 150, 'saturday': 180, 'sunday': 120
        }
        return base_customers.get(day_name, 100)

    def _calculate_staff_requirements(self, predicted_customers: int) -> Dict:
        return {
            'waiters': max(2, predicted_customers // 25),
            'chefs': max(1, predicted_customers // 40),
            'managers': 1 if predicted_customers > 50 else 0
        }

    def _assign_staff(self, staff: List[Staff], requirements: Dict, day_name: str) -> List[Dict]:
        assigned = []

        for role, needed in requirements.items():
            available_staff = [s for s in staff if s.role == role and day_name in s.availability]
            available_staff.sort(key=lambda x: x.efficiency_score, reverse=True)

            for i in range(min(needed, len(available_staff))):
                staff_member = available_staff[i]
                assigned.append({
                    'name': staff_member.name,
                    'role': staff_member.role,
                    'hours': 8,
                    'hourly_rate': staff_member.hourly_rate,
                    'efficiency_score': staff_member.efficiency_score
                })

        return assigned

    def _calculate_schedule_efficiency(self, assigned_staff: List[Dict], requirements: Dict) -> float:
        if not assigned_staff:
            return 0

        avg_efficiency = np.mean([s['efficiency_score'] for s in assigned_staff])
        coverage = len(assigned_staff) / max(1, sum(requirements.values()))

        return min(100, avg_efficiency * coverage * 100)


class CustomerAnalytics:
    def analyze_customer_satisfaction(self, orders: List[Order]) -> Dict:
        rated_orders = [o for o in orders if o.customer_satisfaction is not None]

        if not rated_orders:
            return {'error': 'No satisfaction data available'}

        ratings = [o.customer_satisfaction for o in rated_orders]
        avg_satisfaction = np.mean(ratings)

        satisfaction_by_hour = defaultdict(list)
        for order in rated_orders:
            hour = datetime.fromisoformat(order.order_time).hour
            satisfaction_by_hour[hour].append(order.customer_satisfaction)

        insights = []

        if avg_satisfaction < 3.5:
            insights.append({
                'type': 'concern',
                'message': f'Average satisfaction ({avg_satisfaction:.1f}/5) is below acceptable levels',
                'recommendation': 'Investigate service quality and food preparation times'
            })

        if satisfaction_by_hour:
            best_hour = max(satisfaction_by_hour.keys(),
                            key=lambda h: np.mean(satisfaction_by_hour[h]))
            worst_hour = min(satisfaction_by_hour.keys(),
                             key=lambda h: np.mean(satisfaction_by_hour[h]))

            insights.append({
                'type': 'pattern',
                'message': f'Best performance at {best_hour}:00, worst at {worst_hour}:00',
                'recommendation': 'Analyze staffing and processes during low-satisfaction periods'
            })

        return {
            'average_satisfaction': round(avg_satisfaction, 2),
            'total_ratings': len(rated_orders),
            'satisfaction_distribution': {i: ratings.count(i) for i in range(1, 6)},
            'insights': insights,
            'trend': self._calculate_satisfaction_trend(rated_orders)
        }

    def generate_recommendations(self, orders: List[Order], menu_items: List[MenuItem]) -> List[Dict]:
        recommendations = []

        item_frequency = defaultdict(int)
        for order in orders:
            for item in order.items:
                item_frequency[item['menu_item_id']] += item['quantity']

        combinations = defaultdict(int)
        for order in orders:
            if len(order.items) > 1:
                item_ids = [item['menu_item_id'] for item in order.items]
                for i in range(len(item_ids)):
                    for j in range(i + 1, len(item_ids)):
                        combo = tuple(sorted([item_ids[i], item_ids[j]]))
                        combinations[combo] += 1

        menu_dict = {item.id: item for item in menu_items}

        popular_items = sorted(item_frequency.items(), key=lambda x: x[1], reverse=True)[:3]
        for item_id, frequency in popular_items:
            if item_id in menu_dict:
                recommendations.append({
                    'type': 'popular_item',
                    'item_name': menu_dict[item_id].name,
                    'reason': f'Ordered {frequency} times recently',
                    'confidence': min(100, frequency * 10)
                })

        top_combos = sorted(combinations.items(), key=lambda x: x[1], reverse=True)[:2]
        for combo, frequency in top_combos:
            item1_name = menu_dict.get(combo[0], {}).name if combo[0] in menu_dict else 'Unknown'
            item2_name = menu_dict.get(combo[1], {}).name if combo[1] in menu_dict else 'Unknown'

            recommendations.append({
                'type': 'combo',
                'items': [item1_name, item2_name],
                'reason': f'Frequently ordered together ({frequency} times)',
                'confidence': min(100, frequency * 15)
            })

        return recommendations

    def _calculate_satisfaction_trend(self, rated_orders: List[Order]) -> str:
        if len(rated_orders) < 5:
            return 'insufficient_data'

        recent_ratings = [o.customer_satisfaction for o in rated_orders[-10:]]
        older_ratings = [o.customer_satisfaction for o in rated_orders[-20:-10]] if len(rated_orders) >= 20 else []

        if older_ratings:
            recent_avg = np.mean(recent_ratings)
            older_avg = np.mean(older_ratings)

            if recent_avg > older_avg + 0.2:
                return 'improving'
            elif recent_avg < older_avg - 0.2:
                return 'declining'

        return 'stable'


menu_optimizer = MenuOptimizer()
inventory_ai = InventoryAI()
staff_scheduler = StaffScheduler()
customer_analytics = CustomerAnalytics()


def init_db():
    conn = sqlite3.connect('restaurant_ai.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS menu_items
                 (id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  category TEXT,
                  price REAL,
                  cost REAL,
                  description TEXT,
                  ingredients TEXT,
                  preparation_time INTEGER,
                  popularity_score REAL,
                  profit_margin REAL,
                  dietary_tags TEXT,
                  created_at TEXT,
                  is_available BOOLEAN DEFAULT TRUE)''')

    c.execute('''CREATE TABLE IF NOT EXISTS inventory
                 (id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  category TEXT,
                  current_stock REAL,
                  unit TEXT,
                  reorder_level REAL,
                  supplier TEXT,
                  cost_per_unit REAL,
                  expiry_date TEXT,
                  last_updated TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS orders
                 (id TEXT PRIMARY KEY,
                  table_number INTEGER,
                  items TEXT,
                  total_amount REAL,
                  status TEXT,
                  order_time TEXT,
                  estimated_ready_time TEXT,
                  customer_satisfaction INTEGER,
                  special_notes TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS staff
                 (id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  role TEXT,
                  hourly_rate REAL,
                  efficiency_score REAL,
                  availability TEXT,
                  created_at TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS ai_insights
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  insight_type TEXT,
                  data TEXT,
                  created_at TEXT)''')

    # Check if menu items already exist
    c.execute('SELECT COUNT(*) FROM menu_items')
    count = c.fetchone()[0]

    # Add sample menu items if database is empty
    if count == 0:
        sample_items = [
            ('1', 'Classic Burger', 'main', 12.99, 6.50,
             'Juicy beef patty with lettuce, tomato, onion, and our special sauce', '[]', 15, 0.8, 0.5, '[]',
             datetime.now().isoformat(), True),
            ('2', 'Caesar Salad', 'salad', 8.99, 4.20,
             'Crisp romaine lettuce with parmesan, croutons, and Caesar dressing', '[]', 8, 0.7, 0.53, '["vegetarian"]',
             datetime.now().isoformat(), True),
            ('3', 'Margherita Pizza', 'main', 14.99, 7.80, 'Fresh mozzarella, tomato sauce, and basil on crispy crust',
             '[]', 20, 0.9, 0.48, '["vegetarian"]', datetime.now().isoformat(), True),
            ('4', 'Grilled Salmon', 'main', 18.99, 9.50, 'Fresh Atlantic salmon grilled to perfection with herbs', '[]',
             18, 0.75, 0.5, '["gluten-free"]', datetime.now().isoformat(), True),
            ('5', 'Chocolate Cake', 'dessert', 6.99, 2.80, 'Rich chocolate cake with chocolate frosting', '[]', 5, 0.6,
             0.6, '[]', datetime.now().isoformat(), True),
            (
            '6', 'Tiramisu', 'dessert', 7.99, 3.50, 'Classic Italian dessert with coffee and mascarpone', '[]', 3, 0.65,
            0.56, '["vegetarian"]', datetime.now().isoformat(), True),
            ('7', 'Fresh Orange Juice', 'beverage', 3.99, 1.20, 'Freshly squeezed orange juice', '[]', 2, 0.5, 0.7,
             '["vegan", "gluten-free"]', datetime.now().isoformat(), True),
            ('8', 'Coffee', 'beverage', 2.99, 0.80, 'Premium Colombian coffee', '[]', 3, 0.8, 0.73, '["vegan"]',
             datetime.now().isoformat(), True),
            ('9', 'Caprese Salad', 'appetizer', 9.99, 4.80,
             'Fresh mozzarella, tomatoes, and basil drizzled with balsamic', '[]', 10, 0.55, 0.52,
             '["vegetarian", "gluten-free"]', datetime.now().isoformat(), True),
            (
            '10', 'Chicken Wings', 'appetizer', 11.99, 5.50, 'Crispy chicken wings with your choice of sauce', '[]', 12,
            0.85, 0.54, '[]', datetime.now().isoformat(), True)
        ]

        c.executemany('''INSERT INTO menu_items 
                         (id, name, category, price, cost, description, ingredients,
                          preparation_time, popularity_score, profit_margin, dietary_tags, created_at, is_available)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', sample_items)

    conn.commit()
    conn.close()


@app.route('/api/menu', methods=['GET'])
def get_menu():
    conn = sqlite3.connect('restaurant_ai.db')
    c = conn.cursor()

    c.execute('SELECT * FROM menu_items WHERE is_available = 1')
    menu_items = []

    for row in c.fetchall():
        item = {
            'id': row[0],
            'name': row[1],
            'category': row[2],
            'price': row[3],
            'cost': row[4],
            'description': row[5],
            'ingredients': json.loads(row[6]) if row[6] else [],
            'preparation_time': row[7],
            'popularity_score': row[8],
            'profit_margin': row[9],
            'dietary_tags': json.loads(row[10]) if row[10] else [],
            'created_at': row[11],
            'is_available': bool(row[12])
        }
        menu_items.append(item)

    conn.close()
    return jsonify(menu_items)


@app.route('/api/menu', methods=['POST'])
def add_menu_item():
    data = request.json

    item_id = str(uuid.uuid4())
    profit_margin = (data['price'] - data['cost']) / data['price'] if data['price'] > 0 else 0

    conn = sqlite3.connect('restaurant_ai.db')
    c = conn.cursor()

    c.execute('''INSERT INTO menu_items 
                 (id, name, category, price, cost, description, ingredients,
                  preparation_time, popularity_score, profit_margin, dietary_tags, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (item_id, data['name'], data['category'], data['price'], data['cost'],
               data['description'], json.dumps(data.get('ingredients', [])),
               data['preparation_time'], 0.5, profit_margin,
               json.dumps(data.get('dietary_tags', [])), datetime.now().isoformat()))

    conn.commit()
    conn.close()

    return jsonify({'success': True, 'item_id': item_id})


@app.route('/api/orders', methods=['GET'])
def get_orders():
    conn = sqlite3.connect('restaurant_ai.db')
    c = conn.cursor()

    c.execute('SELECT * FROM orders ORDER BY order_time DESC LIMIT 100')
    orders = []

    for row in c.fetchall():
        order = {
            'id': row[0],
            'table_number': row[1],
            'items': json.loads(row[2]) if row[2] else [],
            'total_amount': row[3],
            'status': row[4],
            'order_time': row[5],
            'estimated_ready_time': row[6],
            'customer_satisfaction': row[7],
            'special_notes': row[8]
        }
        orders.append(order)

    conn.close()
    return jsonify(orders)


@app.route('/api/orders', methods=['POST'])
def create_order():
    data = request.json

    order_id = str(uuid.uuid4())
    order_time = datetime.now()

    total_prep_time = 0
    conn = sqlite3.connect('restaurant_ai.db')
    c = conn.cursor()

    for item in data['items']:
        c.execute('SELECT preparation_time FROM menu_items WHERE id = ?',
                  (item['menu_item_id'],))
        result = c.fetchone()
        if result:
            total_prep_time += result[0] * item['quantity']

    estimated_ready_time = order_time + timedelta(minutes=max(total_prep_time, 15))

    c.execute('''INSERT INTO orders 
                 (id, table_number, items, total_amount, status, order_time,
                  estimated_ready_time, special_notes)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (order_id, data.get('table_number', 0), json.dumps(data['items']),
               data['total_amount'], 'pending', order_time.isoformat(),
               estimated_ready_time.isoformat(), data.get('special_notes', '')))

    conn.commit()
    conn.close()

    return jsonify({
        'success': True,
        'order_id': order_id,
        'estimated_ready_time': estimated_ready_time.isoformat()
    })


@app.route('/api/orders/<order_id>/status', methods=['PUT'])
def update_order_status(order_id):
    data = request.json
    new_status = data.get('status')

    conn = sqlite3.connect('restaurant_ai.db')
    c = conn.cursor()

    c.execute('UPDATE orders SET status = ? WHERE id = ?', (new_status, order_id))

    conn.commit()
    conn.close()

    return jsonify({'success': True})


KIOSK_HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Restaurant Kiosk</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .kiosk-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 20px 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
        }

        .cart-summary {
            display: flex;
            align-items: center;
            gap: 20px;
        }

        .cart-button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 25px;
            border-radius: 15px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .cart-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }

        .cart-count {
            background: #ff4757;
            color: white;
            border-radius: 50%;
            width: 25px;
            height: 25px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            font-weight: bold;
            position: absolute;
            top: -5px;
            right: -5px;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 20px;
            flex: 1;
        }

        .menu-section {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .categories {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }

        .category-btn {
            background: rgba(102, 126, 234, 0.1);
            border: 2px solid transparent;
            color: #667eea;
            padding: 12px 24px;
            border-radius: 25px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            white-space: nowrap;
        }

        .category-btn.active,
        .category-btn:hover {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }

        .menu-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }

        .menu-item {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
            cursor: pointer;
            border: 2px solid transparent;
        }

        .menu-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.15);
            border-color: #667eea;
        }

        .item-name {
            font-size: 1.3rem;
            font-weight: 700;
            color: #333;
            margin-bottom: 8px;
        }

        .item-description {
            color: #666;
            margin-bottom: 15px;
            line-height: 1.5;
        }

        .item-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .item-price {
            font-size: 1.4rem;
            font-weight: 700;
            color: #667eea;
        }

        .add-btn {
            background: linear-gradient(135deg, #2ed573, #7bed9f);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .add-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(46, 213, 115, 0.3);
        }

        .dietary-tags {
            margin-top: 10px;
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }

        .dietary-tag {
            background: #f8f9fa;
            color: #495057;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .cart-panel {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            height: fit-content;
            position: sticky;
            top: 20px;
        }

        .cart-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 20px;
            color: #333;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .cart-items {
            margin-bottom: 20px;
        }

        .cart-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
            margin-bottom: 10px;
        }

        .cart-item-info {
            flex: 1;
        }

        .cart-item-name {
            font-weight: 600;
            margin-bottom: 5px;
        }

        .cart-item-price {
            color: #667eea;
            font-weight: 600;
        }

        .quantity-controls {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .qty-btn {
            background: #667eea;
            color: white;
            border: none;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .qty-btn:hover {
            background: #5a67d8;
            transform: scale(1.1);
        }

        .cart-total {
            border-top: 2px solid #eee;
            padding-top: 20px;
            margin-bottom: 20px;
        }

        .total-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 1.1rem;
        }

        .total-final {
            font-size: 1.3rem;
            font-weight: 700;
            color: #667eea;
            border-top: 1px solid #eee;
            padding-top: 10px;
        }

        .checkout-btn {
            width: 100%;
            background: linear-gradient(135deg, #ff4757, #ff6b7a);
            color: white;
            border: none;
            padding: 15px;
            border-radius: 15px;
            font-size: 1.2rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .checkout-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(255, 71, 87, 0.3);
        }

        .checkout-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .empty-cart {
            text-align: center;
            padding: 40px 20px;
            color: #666;
        }

        .empty-cart i {
            font-size: 3rem;
            margin-bottom: 15px;
            color: #ddd;
        }

        .modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(5px);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }

        .modal-content {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 500px;
            width: 90%;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        .modal-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 20px;
            color: #333;
        }

        .modal-text {
            color: #666;
            margin-bottom: 30px;
            line-height: 1.6;
        }

        .modal-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
        }

        .btn {
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 1rem;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }

        .btn-secondary {
            background: #f8f9fa;
            color: #495057;
            border: 2px solid #dee2e6;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }

        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }

            .cart-panel {
                position: static;
                order: -1;
            }

            .menu-grid {
                grid-template-columns: 1fr;
            }

            .header {
                flex-direction: column;
                text-align: center;
            }

            .categories {
                justify-content: center;
            }
        }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="kiosk-container">
        <div class="header">
            <div class="logo">
                <i class="fas fa-utensils"></i>
                <span>Restaurant Kiosk</span>
            </div>
            <div class="cart-summary">
                <button class="cart-button" onclick="toggleCart()">
                    <i class="fas fa-shopping-cart"></i>
                    <span>Cart</span>
                    <span class="cart-count" id="cart-count" style="display: none;">0</span>
                </button>
            </div>
        </div>

        <div class="main-content">
            <div class="menu-section">
                <div class="categories" id="categories">
                    <button class="category-btn active" onclick="filterCategory('all')">All Items</button>
                </div>

                <div class="menu-grid" id="menu-grid">
                </div>
            </div>

            <div class="cart-panel">
                <h2 class="cart-title">
                    <i class="fas fa-shopping-cart"></i>
                    Your Order
                </h2>

                <div id="cart-content">
                    <div class="empty-cart">
                        <i class="fas fa-shopping-cart"></i>
                        <h3>Your cart is empty</h3>
                        <p>Add some delicious items to get started!</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div id="checkout-modal" class="modal">
        <div class="modal-content">
            <h2 class="modal-title">Confirm Your Order</h2>
            <div id="order-summary" class="modal-text"></div>
            <div class="modal-buttons">
                <button class="btn btn-secondary" onclick="closeModal()">Cancel</button>
                <button class="btn btn-primary" onclick="placeOrder()">
                    <span id="place-order-text">Place Order</span>
                    <span id="place-order-loading" class="loading" style="display: none;"></span>
                </button>
            </div>
        </div>
    </div>

    <div id="success-modal" class="modal">
        <div class="modal-content">
            <h2 class="modal-title">Order Placed Successfully!</h2>
            <div class="modal-text">
                <p><strong>Order Number:</strong> <span id="order-number"></span></p>
                <p><strong>Estimated Ready Time:</strong> <span id="ready-time"></span></p>
            </div>
            <div class="modal-buttons">
                <button class="btn btn-primary" onclick="closeSuccessModal()">Continue Shopping</button>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = '/api';
        let menuItems = [];
        let cart = [];
        let currentCategory = 'all';

        async function loadMenu() {
            try {
                const response = await fetch(`${API_BASE}/menu`);
                const data = await response.json();

                if (Array.isArray(data)) {
                    menuItems = data.filter(item => item.is_available);
                    displayMenu();
                    updateCategories();
                } else {
                    console.error('Invalid menu data:', data);
                    showSampleMenu();
                }
            } catch (error) {
                console.error('Error loading menu:', error);
                showSampleMenu();
            }
        }

        function showSampleMenu() {
            menuItems = [
                {
                    id: '1',
                    name: 'Classic Burger',
                    category: 'main',
                    price: 12.99,
                    description: 'Juicy beef patty with lettuce, tomato, onion, and our special sauce',
                    dietary_tags: [],
                    preparation_time: 15
                },
                {
                    id: '2',
                    name: 'Caesar Salad',
                    category: 'salad',
                    price: 8.99,
                    description: 'Crisp romaine lettuce with parmesan, croutons, and Caesar dressing',
                    dietary_tags: ['vegetarian'],
                    preparation_time: 8
                },
                {
                    id: '3',
                    name: 'Margherita Pizza',
                    category: 'main',
                    price: 14.99,
                    description: 'Fresh mozzarella, tomato sauce, and basil on crispy crust',
                    dietary_tags: ['vegetarian'],
                    preparation_time: 20
                },
                {
                    id: '4',
                    name: 'Chocolate Cake',
                    category: 'dessert',
                    price: 6.99,
                    description: 'Rich chocolate cake with chocolate frosting',
                    dietary_tags: [],
                    preparation_time: 5
                },
                {
                    id: '5',
                    name: 'Fresh Orange Juice',
                    category: 'beverage',
                    price: 3.99,
                    description: 'Freshly squeezed orange juice',
                    dietary_tags: ['vegan', 'gluten-free'],
                    preparation_time: 2
                }
            ];
            displayMenu();
            updateCategories();
        }

        function updateCategories() {
            const categories = [...new Set(menuItems.map(item => item.category))];
            const categoriesContainer = document.getElementById('categories');

            categoriesContainer.innerHTML = `
                <button class="category-btn active" onclick="filterCategory('all')">All Items</button>
                ${categories.map(cat => 
                    `<button class="category-btn" onclick="filterCategory('${cat}')">${capitalizeFirst(cat)}</button>`
                ).join('')}
            `;
        }

        function capitalizeFirst(str) {
            return str.charAt(0).toUpperCase() + str.slice(1);
        }

        function filterCategory(category) {
            currentCategory = category;

            document.querySelectorAll('.category-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');

            displayMenu();
        }

        function displayMenu() {
            const filteredItems = currentCategory === 'all' 
                ? menuItems 
                : menuItems.filter(item => item.category === currentCategory);

            const menuGrid = document.getElementById('menu-grid');

            menuGrid.innerHTML = filteredItems.map(item => `
                <div class="menu-item" onclick="addToCart('${item.id}')">
                    <div class="item-name">${item.name}</div>
                    <div class="item-description">${item.description}</div>
                    <div class="item-footer">
                        <div class="item-price">${item.price.toFixed(2)}</div>
                        <button class="add-btn">
                            <i class="fas fa-plus"></i> Add
                        </button>
                    </div>
                    ${item.dietary_tags && item.dietary_tags.length > 0 ? `
                        <div class="dietary-tags">
                            ${item.dietary_tags.map(tag => `<span class="dietary-tag">${tag}</span>`).join('')}
                        </div>
                    ` : ''}
                </div>
            `).join('');
        }

        function addToCart(itemId) {
            const item = menuItems.find(i => i.id === itemId);
            if (!item) return;

            const existingItem = cart.find(i => i.id === itemId);

            if (existingItem) {
                existingItem.quantity += 1;
            } else {
                cart.push({
                    id: itemId,
                    name: item.name,
                    price: item.price,
                    quantity: 1
                });
            }

            updateCartDisplay();
            updateCartCount();
        }

        function removeFromCart(itemId) {
            const itemIndex = cart.findIndex(i => i.id === itemId);
            if (itemIndex > -1) {
                cart.splice(itemIndex, 1);
                updateCartDisplay();
                updateCartCount();
            }
        }

        function updateQuantity(itemId, change) {
            const item = cart.find(i => i.id === itemId);
            if (!item) return;

            item.quantity += change;

            if (item.quantity <= 0) {
                removeFromCart(itemId);
            } else {
                updateCartDisplay();
                updateCartCount();
            }
        }

        function updateCartCount() {
            const totalItems = cart.reduce((sum, item) => sum + item.quantity, 0);
            const cartCount = document.getElementById('cart-count');

            if (totalItems > 0) {
                cartCount.textContent = totalItems;
                cartCount.style.display = 'flex';
            } else {
                cartCount.style.display = 'none';
            }
        }

        function updateCartDisplay() {
            const cartContent = document.getElementById('cart-content');

            if (cart.length === 0) {
                cartContent.innerHTML = `
                    <div class="empty-cart">
                        <i class="fas fa-shopping-cart"></i>
                        <h3>Your cart is empty</h3>
                        <p>Add some delicious items to get started!</p>
                    </div>
                `;
                return;
            }

            const subtotal = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
            const tax = subtotal * 0.08;
            const total = subtotal + tax;

            cartContent.innerHTML = `
                <div class="cart-items">
                    ${cart.map(item => `
                        <div class="cart-item">
                            <div class="cart-item-info">
                                <div class="cart-item-name">${item.name}</div>
                                <div class="cart-item-price">${item.price.toFixed(2)} each</div>
                            </div>
                            <div class="quantity-controls">
                                <button class="qty-btn" onclick="updateQuantity('${item.id}', -1)">-</button>
                                <span>${item.quantity}</span>
                                <button class="qty-btn" onclick="updateQuantity('${item.id}', 1)">+</button>
                            </div>
                        </div>
                    `).join('')}
                </div>

                <div class="cart-total">
                    <div class="total-row">
                        <span>Subtotal:</span>
                        <span>${subtotal.toFixed(2)}</span>
                    </div>
                    <div class="total-row">
                        <span>Tax (8%):</span>
                        <span>${tax.toFixed(2)}</span>
                    </div>
                    <div class="total-row total-final">
                        <span>Total:</span>
                        <span>${total.toFixed(2)}</span>
                    </div>
                </div>

                <button class="checkout-btn" onclick="showCheckout()">
                    <i class="fas fa-credit-card"></i>
                    Checkout - ${total.toFixed(2)}
                </button>
            `;
        }

        function showCheckout() {
            if (cart.length === 0) return;

            const subtotal = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
            const tax = subtotal * 0.08;
            const total = subtotal + tax;

            const orderSummary = document.getElementById('order-summary');
            orderSummary.innerHTML = `
                <div style="text-align: left; margin-bottom: 20px;">
                    ${cart.map(item => `
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <span>${item.name} x${item.quantity}</span>
                            <span>${(item.price * item.quantity).toFixed(2)}</span>
                        </div>
                    `).join('')}
                    <hr style="margin: 15px 0;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Subtotal:</span>
                        <span>${subtotal.toFixed(2)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                        <span>Tax:</span>
                        <span>${tax.toFixed(2)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-weight: bold; font-size: 1.1rem;">
                        <span>Total:</span>
                        <span>${total.toFixed(2)}</span>
                    </div>
                </div>
            `;

            document.getElementById('checkout-modal').style.display = 'flex';
        }

        async function placeOrder() {
            const placeOrderText = document.getElementById('place-order-text');
            const placeOrderLoading = document.getElementById('place-order-loading');

            placeOrderText.style.display = 'none';
            placeOrderLoading.style.display = 'inline-block';

            const orderData = {
                items: cart.map(item => ({
                    menu_item_id: item.id,
                    quantity: item.quantity,
                    special_requests: ''
                })),
                total_amount: cart.reduce((sum, item) => sum + (item.price * item.quantity), 0) * 1.08,
                special_notes: ''
            };

            try {
                const response = await fetch(`${API_BASE}/orders`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(orderData)
                });

                const result = await response.json();

                if (result.success) {
                    document.getElementById('order-number').textContent = result.order_id.substring(0, 8).toUpperCase();
                    document.getElementById('ready-time').textContent = new Date(result.estimated_ready_time).toLocaleTimeString();

                    closeModal();
                    document.getElementById('success-modal').style.display = 'flex';

                    cart = [];
                    updateCartDisplay();
                    updateCartCount();
                } else {
                    throw new Error('Order failed');
                }

            } catch (error) {
                console.error('Error placing order:', error);
                alert('Sorry, there was an error placing your order. Please try again.');
            }

            placeOrderText.style.display = 'inline';
            placeOrderLoading.style.display = 'none';
        }

        function closeModal() {
            document.getElementById('checkout-modal').style.display = 'none';
        }

        function closeSuccessModal() {
            document.getElementById('success-modal').style.display = 'none';
        }

        function toggleCart() {
        }

        document.addEventListener('DOMContentLoaded', () => {
            loadMenu();
        });
    </script>
</body>
</html>'''


@app.route('/')
def serve_kiosk():
    return render_template_string(KIOSK_HTML)


if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5002)