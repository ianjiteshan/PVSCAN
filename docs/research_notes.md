
# Backend Research Notes

## Flask API Best Practices (Auth0 Guide)

### API Design Principles
1. **Resource Naming:**
   - Use nouns in plural form: `/users`, `/users/{userId}`
   - Use hyphens for readability: `/mobile-devices`
   - Use forward slashes for hierarchy: `/users/{userId}/orders`
   - Use lowercase letters only

2. **HTTP Verbs:**
   - GET: data retrieval
   - POST: create new resource
   - PUT: update entire resource
   - DELETE: delete resource
   - PATCH: partial update

### Application Structure
```
project/
    api/
        model/          # Data descriptors/database models
        route/          # URI definitions and endpoints
        schema/         # Input/output definitions
        service/        # Business logic and external interactions
    test/
        route/
    app.py
    requirements.txt
```

### Key Principles:
- Routes should be simple and delegate logic to services
- Separate concerns: models, routes, schemas, services
- Use blueprints for modular organization
- Implement proper error handling and validation



## Vue.js Best Practices (Vue School Guide)

### Core Principle: Predictability
- Ability to intuitively go from feature request to code location
- Quick understanding of available tools at any location
- Makes onboarding easier and development more efficient

### Community Standards Sources:
1. Vue.js Style Guide
2. Vue CLI scaffolding
3. Official Vue.js libraries (Vue Router, Pinia/Vuex)
4. Popular component frameworks (Vuetify, Quasar)

### Component Naming Conventions:
- Single File Components in PascalCase
- Base components with same prefix (Base/App)
- Single instance components with "The" prefix
- Tightly coupled child components prefixed with parent name
- Names should go from general to specific (SearchWidgetInput)

### Project Structure Principles:
- Start with Vue CLI generated structure
- Don't overthink or deviate without good reason
- Use official libraries for standardization
- Organize by feature when possible
- Separate concerns clearly


## Dashboard Design Best Practices (Sisense Guide)

### Core Design Goals:
- **Make complex simple**: Simplify lots of changing data and analytical needs
- **Tell clear story**: Connect data to business context and answer viewer questions
- **Express data meaning**: Visualizations must correctly represent the data
- **Reveal details as needed**: Right level of detail for each user type

### 4 Key Design Principles:

#### 1. The 5 Second Rule
- Dashboard should provide relevant information in ~5 seconds
- Most important metrics should immediately pop from screen
- Frequently asked business questions answered at a glance

#### 2. Logical Layout: Inverted Pyramid
- **Top**: Most significant insights (headlines)
- **Middle**: Trends and context (supporting details)
- **Bottom**: Granular details for drilling down (background info)

#### 3. Minimalism: Less Is More
- Maximum 5-9 visualizations per dashboard
- Human brain can only comprehend 7±2 images at once
- Use filters and hierarchies instead of cramming more widgets
- Break into multiple dashboards if needed

#### 4. Right Data Visualization
Choose visualization based on purpose:
- **Relationship**: Connection between variables
- **Comparison**: Side-by-side variable comparison
- **Composition**: Breaking data into components
- **Distribution**: Range and grouping of values

### Additional Considerations:
- Design for specific end-user needs
- Gather requirements and define KPIs first
- Consider user role and daily workflow
- Beautiful design means nothing without decision-making impact

