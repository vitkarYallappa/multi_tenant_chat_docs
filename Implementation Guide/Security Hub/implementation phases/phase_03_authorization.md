# Phase 3: Authorization & RBAC System
**Duration**: Week 4 (7 days)  
**Team**: 2-3 developers  
**Dependencies**: Phase 2 (Authentication system)  

## Overview
Implement comprehensive Role-Based Access Control (RBAC) system with permission management, policy evaluation, resource-level authorization, and fine-grained access control.

## Step 9: RBAC Engine Implementation

### New Folders/Files to Create
```
src/
├── core/
│   ├── authz/
│   │   ├── __init__.py
│   │   ├── rbac_engine.py
│   │   ├── permission_evaluator.py
│   │   ├── policy_engine.py
│   │   ├── resource_guard.py
│   │   └── role_manager.py
├── models/
│   ├── postgres/
│   │   ├── permission_model.py
│   │   ├── role_model.py
│   │   └── policy_model.py
│   └── domain/
│       ├── permission_models.py
│       └── authorization_models.py
├── services/
│   └── authorization_service.py
├── api/v2/
│   ├── permission_routes.py
│   └── role_routes.py
```

### Core Authorization Components

#### `/src/core/authz/rbac_engine.py`
**Purpose**: Core Role-Based Access Control logic and permission resolution  
**Technology**: Complex permission inheritance, caching, hierarchical roles  

**Classes & Methods**:
- `RBACEngine`: Main authorization engine
  - `check_permission(user_context, resource, action, resource_attrs)`: Permission validation
    - Parameters: UserContext, resource (str), action (str), resource_attrs (Dict)
    - Returns: PermissionResult (allowed, reason, applied_policies)
  - `get_user_permissions(user_id, tenant_id, resource_type)`: Aggregate user permissions
    - Parameters: user_id (str), tenant_id (str), resource_type (Optional[str])
    - Returns: UserPermissionSet with resolved permissions
  - `resolve_role_hierarchy(role, tenant_id)`: Calculate inherited permissions
    - Parameters: role (str), tenant_id (str)
    - Returns: HierarchyResult with inheritance chain
  - `evaluate_permission_expression(expression, context)`: Dynamic permission evaluation
    - Parameters: expression (str), context (EvaluationContext)
    - Returns: bool (evaluation result)
  - `check_batch_permissions(user_context, permission_requests)`: Bulk permission checking
    - Parameters: UserContext, permission_requests (List[PermissionRequest])
    - Returns: BatchPermissionResult
  - `get_accessible_resources(user_context, resource_type, filters)`: Resource discovery
  - `audit_permission_check(user_context, resource, action, result)`: Audit logging

**Performance Features**:
- Permission caching with TTL, batch permission evaluation
- Lazy loading of role hierarchies, optimized database queries

#### `/src/core/authz/permission_evaluator.py`
**Purpose**: Complex permission logic evaluation and context analysis  
**Technology**: Expression parsing, attribute-based evaluation, policy composition  

**Classes & Methods**:
- `PermissionEvaluator`: Permission evaluation engine
  - `evaluate_permission_policy(policy, context, resource_attrs)`: Policy evaluation
    - Parameters: policy (PolicyDefinition), context (UserContext), resource_attrs (Dict)
    - Returns: PolicyEvaluationResult with decision and reasoning
  - `evaluate_attribute_conditions(conditions, context)`: Attribute-based evaluation
    - Parameters: conditions (List[Condition]), context (EvaluationContext)
    - Returns: ConditionResult
  - `evaluate_time_constraints(constraints, current_time)`: Time-based access control
  - `evaluate_resource_ownership(user_context, resource_id)`: Ownership validation
  - `evaluate_tenant_isolation(user_context, resource_tenant)`: Multi-tenant security
  - `evaluate_data_classification_access(user_clearance, data_classification)`: Data access
  - `combine_policy_results(results, combination_logic)`: Multiple policy resolution
  - `get_evaluation_trace(evaluation_id)`: Debugging and audit

**Advanced Features**:
- Dynamic condition evaluation, contextual permission modification
- Delegation and temporary permissions, conditional access grants

#### `/src/core/authz/policy_engine.py`
**Purpose**: Policy definition, management, and evaluation framework  
**Technology**: Policy DSL, JSON-based policies, dynamic policy updates  

**Classes & Methods**:
- `PolicyEngine`: Policy management and evaluation
  - `load_policies(tenant_id, resource_type)`: Dynamic policy loading
    - Parameters: tenant_id (str), resource_type (Optional[str])
    - Returns: PolicySet with compiled policies
  - `compile_policy(policy_definition)`: Policy compilation and optimization
    - Parameters: policy_definition (Dict)
    - Returns: CompiledPolicy
  - `evaluate_policy_set(policies, context, resource)`: Multiple policy evaluation
    - Parameters: policies (PolicySet), context (UserContext), resource (Resource)
    - Returns: PolicySetResult
  - `update_policy(policy_id, policy_definition)`: Runtime policy updates
  - `validate_policy_syntax(policy_definition)`: Policy validation
  - `get_applicable_policies(context, resource)`: Policy filtering
  - `cache_policy_decisions(cache_key, decision, ttl)`: Decision caching
  - `audit_policy_evaluation(policy_id, context, result)`: Evaluation tracking

**Policy Features**:
- JSON-based policy language, inheritance and composition
- Runtime policy updates, validation and testing

#### `/src/core/authz/resource_guard.py`
**Purpose**: Resource-level access control and data filtering  
**Technology**: Query modification, field-level security, data masking  

**Classes & Methods**:
- `ResourceGuard`: Resource protection and access control
  - `apply_resource_filters(user_context, query, resource_type)`: Query filtering
    - Parameters: UserContext, query (DatabaseQuery), resource_type (str)
    - Returns: FilteredQuery with access controls
  - `apply_field_level_security(user_context, data, classification)`: Field filtering
    - Parameters: UserContext, data (Dict), classification (DataClassification)
    - Returns: FilteredData with masked/removed fields
  - `check_resource_ownership(user_id, resource_id, resource_type)`: Ownership validation
  - `apply_tenant_isolation(tenant_id, query)`: Multi-tenant filtering
  - `get_user_accessible_resources(user_context, resource_type)`: Resource listing
  - `apply_data_classification_filters(user_clearance, data)`: Classification-based filtering
  - `generate_access_audit_log(user_context, resource, access_type)`: Access logging
  - `validate_bulk_access(user_context, resource_ids)`: Bulk authorization

**Security Features**:
- Automatic tenant isolation, field-level data protection
- Ownership-based access control, audit trail generation

## Step 10: Permission & Role Management

#### `/src/models/postgres/permission_model.py`
**Purpose**: Permission and role data models for PostgreSQL  
**Technology**: SQLAlchemy, hierarchical data, JSON configurations  

**Models**:
- `Permission`: Individual permission definition
  - Fields: permission_id, name, description, resource_type, action, conditions
  - Methods: `validate_syntax()`, `get_applicable_resources()`, `check_conflicts()`

- `Role`: Role definition with permission associations
  - Fields: role_id, name, description, permissions, inherits_from, tenant_specific
  - Methods: `get_effective_permissions()`, `validate_hierarchy()`, `get_users()`

- `RoleAssignment`: User-role assignments with context
  - Fields: user_id, role_id, tenant_id, context, granted_by, expires_at
  - Methods: `is_active()`, `get_context_permissions()`, `check_delegation()`

- `PolicyDefinition`: Stored policy configurations
  - Fields: policy_id, name, type, definition, priority, status
  - Methods: `compile()`, `validate()`, `get_evaluation_history()`

**Relationships**: Many-to-many with junction tables, hierarchical role inheritance

#### `/src/models/domain/authorization_models.py`
**Purpose**: Pydantic models for authorization operations  
**Technology**: Pydantic v2, validation, serialization  

**Models**:
- `PermissionRequest`: Permission check request
  - Fields: resource (str), action (str), resource_attributes (Dict), context (Dict)
  - Validators: Resource format validation, action enumeration

- `PermissionResult`: Permission check response
  - Fields: allowed (bool), reason (str), applied_policies (List), conditions (List)
  - Methods: `to_audit_log()`, `get_debug_info()`

- `UserContext`: User authorization context
  - Fields: user_id, tenant_id, roles, permissions, attributes, session_info
  - Methods: `has_permission()`, `get_role_permissions()`, `is_admin()`

- `PolicyEvaluationContext`: Policy evaluation environment
  - Fields: user_context, resource_attributes, environment_attributes, time_context
  - Methods: `get_attribute()`, `evaluate_expression()`, `add_context()`

**Validation Features**: Input sanitization, format validation, business rule enforcement

#### `/src/services/authorization_service.py`
**Purpose**: High-level authorization service orchestration  
**Technology**: Service composition, caching, audit integration  

**Classes & Methods**:
- `AuthorizationService`: Main authorization orchestrator
  - `authorize_action(user_context, resource, action, resource_attrs)`: Main authorization
    - Parameters: UserContext, resource (str), action (str), resource_attrs (Dict)
    - Returns: AuthorizationResult with detailed decision
  - `batch_authorize(user_context, requests)`: Bulk authorization
    - Parameters: UserContext, requests (List[PermissionRequest])
    - Returns: BatchAuthorizationResult
  - `get_user_permissions_summary(user_id, tenant_id)`: Permission overview
  - `check_administrative_access(user_context, admin_action)`: Admin authorization
  - `authorize_data_access(user_context, data_query, classification)`: Data access control
  - `delegate_permission(delegator_context, delegatee_id, permissions, duration)`: Delegation
  - `revoke_delegated_permissions(delegator_context, delegation_id)`: Revocation
  - `audit_authorization_decision(context, request, result)`: Audit logging

**Business Logic**:
- Permission aggregation and caching, delegation management
- Audit trail generation, performance optimization

## Step 11: Authorization API Endpoints

#### `/src/api/v2/permission_routes.py`
**Purpose**: Permission management REST API  
**Technology**: FastAPI, OAuth2 security, input validation  

**Endpoints**:
- `POST /permissions/check`: Single permission check
  - Request: PermissionCheckRequest (resource, action, context)
  - Response: PermissionResult
  - Security: Requires valid authentication

- `POST /permissions/batch-check`: Bulk permission checking
  - Request: BatchPermissionRequest (requests list)
  - Response: BatchPermissionResult
  - Performance: Optimized for bulk operations

- `GET /permissions/user/{user_id}`: User permission summary
  - Parameters: user_id (path), resource_type (query)
  - Response: UserPermissionSummary
  - Security: Admin access or self-access only

- `POST /permissions/delegate`: Permission delegation
  - Request: DelegationRequest (delegatee, permissions, duration)
  - Response: DelegationResult
  - Security: Delegation authorization required

- `GET /permissions/effective`: Current user's effective permissions
  - Response: EffectivePermissionSet
  - Usage: UI permission-based rendering

#### `/src/api/v2/role_routes.py`
**Purpose**: Role management REST API  
**Technology**: FastAPI, admin authorization, role validation  

**Endpoints**:
- `GET /roles/`: List available roles
  - Query Parameters: tenant_id, include_system_roles
  - Response: RoleList with descriptions
  - Security: Role management permission required

- `POST /roles/`: Create new role
  - Request: CreateRoleRequest (name, description, permissions)
  - Response: RoleCreationResult
  - Validation: Permission conflicts, naming conventions

- `PUT /roles/{role_id}`: Update role definition
  - Parameters: role_id (path)
  - Request: UpdateRoleRequest
  - Security: Role modification permission

- `POST /roles/{role_id}/assign`: Assign role to user
  - Parameters: role_id (path)
  - Request: RoleAssignmentRequest (user_id, context, duration)
  - Response: AssignmentResult

- `DELETE /roles/{role_id}/assignments/{assignment_id}`: Revoke role assignment
  - Parameters: role_id, assignment_id
  - Security: Assignment management permission

## Cross-Service Integration

### Authentication Integration
- **Token Validation**: Integration with JWT validation
- **User Context**: Enrichment with authorization data
- **Session Enhancement**: Permission caching in sessions

### Database Integration
- **Permission Queries**: Optimized database access patterns
- **Role Hierarchies**: Efficient hierarchy resolution
- **Audit Logging**: Authorization decision tracking

### Other Services Integration
- **Chat Service**: Message access authorization
- **MCP Engine**: Flow execution permissions
- **Analytics Engine**: Data access authorization
- **Model Orchestrator**: API usage permissions

## Performance Optimizations

### Caching Strategy
- **Permission Cache**: User permission sets (5-minute TTL)
- **Role Cache**: Role definitions (1-hour TTL)
- **Policy Cache**: Compiled policies (30-minute TTL)
- **Decision Cache**: Authorization decisions (1-minute TTL)

### Database Optimization
- **Index Strategy**: Optimized queries for permission checks
- **Batch Operations**: Bulk permission evaluation
- **Connection Pooling**: Shared database connections
- **Query Optimization**: Efficient role hierarchy queries

### Evaluation Optimization
- **Short-Circuit Evaluation**: Early denial/approval
- **Parallel Evaluation**: Concurrent policy evaluation
- **Compiled Policies**: Pre-compiled policy expressions
- **Lazy Loading**: On-demand resource loading

## Security Considerations

### Authorization Security
- **Fail-Safe Defaults**: Deny by default approach
- **Privilege Escalation Prevention**: Role hierarchy validation
- **Tenant Isolation**: Strict tenant boundary enforcement
- **Audit Trail**: Comprehensive authorization logging

### Data Protection
- **Field-Level Security**: Sensitive data protection
- **Classification-Based Access**: Data sensitivity handling
- **Ownership Validation**: Resource ownership checks
- **Query Modification**: Automatic security filtering

### Policy Security
- **Policy Validation**: Syntax and logic validation
- **Safe Policy Updates**: Rollback capabilities
- **Policy Testing**: Sandbox evaluation environment
- **Version Control**: Policy change tracking

## Error Handling

### Authorization Errors
- **Permission Denied**: Clear error messages without information leakage
- **Invalid Resource**: Resource not found vs. access denied
- **Policy Errors**: Policy evaluation failures
- **System Errors**: Graceful degradation

### Performance Errors
- **Timeout Handling**: Authorization timeout management
- **Cache Failures**: Fallback to database queries
- **Database Errors**: Service degradation modes
- **Circuit Breaker**: Protection against cascading failures

## Monitoring & Auditing

### Authorization Metrics
- Permission check success/failure rates
- Authorization decision latency
- Policy evaluation performance
- Cache hit/miss ratios

### Security Metrics
- Permission denial patterns
- Administrative action monitoring
- Privilege escalation attempts
- Suspicious authorization patterns

### Audit Requirements
- All authorization decisions logged
- Permission changes tracked
- Role assignments/revocations recorded
- Policy modifications audited

## Testing Strategy

### Unit Tests
- RBAC engine permission evaluation
- Policy engine rule processing
- Permission evaluator logic
- Resource guard filtering

### Integration Tests
- End-to-end authorization flows
- Cross-service permission checking
- Role assignment workflows
- Policy update procedures

### Security Tests
- Privilege escalation prevention
- Tenant isolation validation
- Authorization bypass attempts
- Performance under load

## Success Criteria
- [ ] RBAC engine fully operational with hierarchical roles
- [ ] Permission evaluation system working with complex policies
- [ ] Resource-level authorization implemented
- [ ] API endpoints for permission and role management
- [ ] Cross-service authorization integration complete
- [ ] Performance targets met (sub-100ms for simple checks)
- [ ] Security audit logging operational
- [ ] Caching system providing performance benefits
- [ ] Comprehensive test coverage (>90% for authorization logic)