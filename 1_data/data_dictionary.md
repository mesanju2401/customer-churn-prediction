
### 2. 1_data/data_dictionary.md

```markdown
# Data Dictionary - Customer Churn Dataset

## Source Information
- **Dataset**: mstz/churn
- **URL**: https://huggingface.co/datasets/mstz/churn
- **License**: Apache 2.0
- **Last Updated**: 2023
- **Records**: 3,333 customers

## Feature Descriptions

### Customer Demographics
- **customerID**: Unique identifier for each customer (string)
- **gender**: Customer gender (Male/Female)
- **SeniorCitizen**: Whether customer is 65+ years (0/1)
- **Partner**: Whether customer has a partner (Yes/No)
- **Dependents**: Whether customer has dependents (Yes/No)

### Account Information
- **tenure**: Number of months customer has stayed with company (integer)
- **Contract**: Contract term (Month-to-month, One year, Two year)
- **PaperlessBilling**: Whether customer has paperless billing (Yes/No)
- **PaymentMethod**: Payment method (Electronic check, Mailed check, Bank transfer, Credit card)
- **MonthlyCharges**: Monthly amount charged to customer (float)
- **TotalCharges**: Total amount charged to customer (float)

### Services
- **PhoneService**: Whether customer has phone service (Yes/No)
- **MultipleLines**: Whether customer has multiple lines (Yes/No/No phone service)
- **InternetService**: Internet service provider (DSL, Fiber optic, No)
- **OnlineSecurity**: Whether customer has online security (Yes/No/No internet service)
- **OnlineBackup**: Whether customer has online backup (Yes/No/No internet service)
- **DeviceProtection**: Whether customer has device protection (Yes/No/No internet service)
- **TechSupport**: Whether customer has tech support (Yes/No/No internet service)
- **StreamingTV**: Whether customer has streaming TV (Yes/No/No internet service)
- **StreamingMovies**: Whether customer has streaming movies (Yes/No/No internet service)

### Target Variable
- **Churn**: Whether customer churned (Yes/No) - Binary target variable

## Data Quality Notes
- **TotalCharges**: Contains some empty strings that need to be converted to numeric
- **tenure**: Ranges from 0-72 months
- **MonthlyCharges**: Ranges from $18.25 to $118.75
- **Missing Values**: TotalCharges has 11 missing values (empty strings)

## Feature Engineering Opportunities
1. **tenure_group**: Categorize tenure into bins (0-12, 12-24, 24-48, 48+)
2. **avg_monthly_charge**: TotalCharges / tenure (for tenure > 0)
3. **services_count**: Total number of services subscribed
4. **has_streaming**: Whether customer has any streaming service
5. **payment_type_risk**: High-risk payment methods (Electronic check)