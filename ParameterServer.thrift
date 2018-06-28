service ParameterServer {
    string push(1: i32 wid, 2: string key, 3: string value_json, 4: i16 time_stamp)
    string pull(1: i32 wid, 2: string key, 3: i16 time_stamp)
    string init(1: i32 wid, 2: string key, 3: string init_value_json)
    i32 register_worker()
    string goodbye(1: i32 wid)
}
